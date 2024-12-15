from collections import defaultdict
import math
import gymnasium as gym
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, LineString, Point
from shapely.strtree import STRtree
from PIL import Image, ImageDraw
# import pyproj as proj
from pyproj.crs import ProjectedCRS
import pygame
from pygame import gfxdraw
import random
import torch

class CoordProj:
    def __init__(self, lon, lat):
#         laea = LambertAzimuthalEqualAreaConversion(lat, lon)
#         self.crs = ProjectedCRS(conversion=laea)
        self.crs_wgs = proj.Proj(init='epsg:4326', long_0=lon, lat_0=lat)
        self.crs_bng = proj.Proj(init='epsg:21480')
        
    def get_local(self, lon, lat):
        return proj.transform(self.crs_wgs, self.crs_bng, lon, lat)
    
    def get_wgs(self, x, y):
        return proj.transform(self.crs_bng, self.crs_wgs, x, y)
    
    def get_locals(self, arr):
        return [(*self.get_local(lon, lat),z) for (lon, lat, z) in arr]
    
    def get_wgss(self, arr):
        ret = []
        for coord in arr:
            ret.append(self.get_wgs(coord[0], coord[1]))
        return ret
    
    def get_crs(self):
        return self.crs_bng

class BreakLineInfo():
    def __init__(self, break_points):
        self.reward = 0
        self.delta = 0
        self.complete = False
        self.bpis = []

    def is_complete(self):
        return self.complete

    def add_reward(self, reward):
        self.reward = reward
        self.complete = True

    def add_dalta_group(self, delta):
        # 增加的车道组数量
        self.delta = delta

class BreakEnv():
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, name, render_mode = 'rgb_array'):
        self.L = 25    # 可视范围，m
        self.render_model = render_mode
        self.speed = 1.0
        self.max_distance = 0.5
        self.name = name
        self.surf = None
        self.screen = None
        self.clock = None
        # self.projector = CoordProj(113.261934, 22.853765)
        self.ref_coord = [0, 0]
        self.frame_size = [0, 0]    # frame size with pixel unit

        self.dividers = dict()
        self.das = dict()
        self.nodes = dict()
        self.track = list()
        # 保存关键点id和信息
        self.keypoints = dict()
        self.keypoints_geo = []
        # 关键点四叉树
        self.keypoint_tree = None
         # 保存div_node跟下一个node的映射关系
        self.node_next_node_map = defaultdict(list)
         # 保存div_node跟上一个node的映射关系
        self.node_prev_node_map = defaultdict(list)
        # 保存原始node跟div的映射关系
        self.node_div_map = defaultdict(set)
        # 所有主辅线的长度总和
        self.total_len = 0
        # 所有形点构成的线段
        self.segments = dict()
        self.segments_str = defaultdict()
        # 所有形点线段构成的四叉树
        self.segm_tree = None
        # 所有形点线段的拓扑连接关系
        self.segm_topo = dict()
        # 形点线段对应的出nodes映射
        self.seg_next_node_map = defaultdict(set)
        # node进入的形点线段映射
        self.node_next_seg_map = defaultdict(set)
        # node连通的所有node映射(提升查找性能)
        self.node_prev_nodes = defaultdict(set)
        
        self._clean_()
        
        if name == 'Break-v1':  # 简单收费站场景
            self.version = 1
            self._load_v1_()
        elif name == 'Break-v2': # 收费站+高速混合场景
            self.version = 2
        elif name == 'Break-v3': # 收费站+高速+高架混合场景
            self.version = 3
        elif name == 'Break-v4': # 城市场景
            self.version = 4
        # self._proj_center_()

    @property
    def observation_space(self):
        '''
        start_node_id, end_node_id, main_aux, virtual, color, type, group type, material, width, overlay, start_x, start_y, start_z, end_x, end_y, end_z
        '''
        return np.array([['start_node_id'], ['end_node_id'], ['main_aux'], ['virtual'], ['color'], ['type'], ['group type'], ['material'], ['width'], ['overlay'], ['start_x'], ['start_y'], ['start_z'], ['end_x'], ['end_y'], ['end_z']])

    @property
    def action_space(self):
        '''
        [
            [start_node_id, end_node_id, t]
        ]
        '''
        return np.array([
            ['start_node_id', 'end_node_id', 't']
        ])

    @property
    def render_mode(self):
        return self.render_model

    def _get_coord_str_(self, coord):
        return '{},{},{}'.format(coord.x, coord.y, coord.z)

    def _get_center_(self, nodes):
        x = 0.
        y = 0.
        cnt = 0
        for _, row in nodes.iterrows():
            x += float(row['geometry'].x)
            y += float(row['geometry'].y)
            cnt += 1

        cx = x / cnt
        cy = y / cnt
        for _, row in nodes.iterrows():
            hw = abs(float(row['geometry'].x) - cx)
            hh = abs(float(row['geometry'].y) - cy)
            if hw > self.frame_size[0]:
                self.frame_size[0] = 2 * hw
            if hh > self.frame_size[1]:
                self.frame_size[1] = 2 * hh
        return (cx, cy)

    def _proj_center_(self, nodes):
        for _, node in nodes.iterrows():
            pt = node['geometry']
            pt.x -= self.ref_coord[0]
            pt.y -= self.ref_coord[1]

    def _load_v1_(self):
        dividers = gpd.read_file('tollgate/DIVIDER.shp').set_crs('epsg:4326')
        das = gpd.read_file('tollgate/DIVIDER_ATTRIBUTE.dbf')
        div_nodes = gpd.read_file('tollgate/DIVIDER_NODE.shp').set_crs('epsg:4326')
        # 构建坐标到node映射
        coord_node_map = defaultdict(int)
        for _, row in div_nodes.iterrows():
            pt = row['geometry']
            k = self._get_coord_str_(pt)
            coord_node_map[k] = int(row['ID'])
        
        node_in_map = defaultdict(list)
        node_out_map = defaultdict(list)
        # 初始化div及其按顺序存储的node id
        div_nodes_map = defaultdict(list)
        for _, div in dividers.iterrows():
            div_id = div['ID']
            linestring = div['geometry']
            for pt in linestring.coords:
                k = str(pt[0]) + ',' + str(pt[1]) + ',' + str(pt[2])
                div_nodes_map[div_id].append(coord_node_map[k])
            
            # div内部节点构建前去后继关系
            for i in range(len(div_nodes_map[div_id]) - 1):
                next_id = div_nodes_map[div_id][i + 1]
                prev_id = div_nodes_map[div_id][i]
                self.node_next_node_map[prev_id].append(next_id)
                self.node_prev_node_map[next_id].append(prev_id)
                self.node_div_map[prev_id].add(int(div_id))
                self.node_div_map[next_id].add(int(div_id))

        for _, div in dividers.iterrows():
            div_id = div['ID']
            ns = div_nodes_map[div_id]
            node_out_map[ns[0]].append(div_id)
            node_in_map[ns[len(ns) - 1]].append(div_id)

        # 转坐标系
        nodes = div_nodes.to_crs(21480)
        dividers = dividers.to_crs(21480)

        # 设置参考中心
        self.ref_coord = self._get_center_(nodes)
        # self._proj_center_(nodes)
        self.org_div_nodes = nodes

        for _, row in das.iterrows():
            div_id = int(row['DIVIDER_ID'])
            self.das[div_id] = row

        for _, row in dividers.iterrows():
            coords = []
            for coord in row['geometry'].coords:
                coords.append((coord[0] - self.ref_coord[0], coord[1] - self.ref_coord[1]))
            linstr = LineString(coords)
            row['geometry'] = linstr
            div_id = int(row['ID'])
            self.dividers[div_id] = row
            # 计算总车道线(排除辅线)长度
            if self.das[div_id]['MAIN_AUX'] != 2:
                self.total_len += linstr.length

        for _, row in self.org_div_nodes.iterrows():
            org_coord = row['geometry']
            coord = [org_coord.x - self.ref_coord[0], org_coord.y - self.ref_coord[1], org_coord.z]
            row['geometry'] = Point(coord)
            self.nodes[int(row['ID'])] = row

        # 初始化node前趋连通关系,车道分离合并或者断头处结束
        for node_id in self.nodes:
            # 前趋
            cur_nid = node_id
            while len(self.node_prev_node_map[cur_nid]) == 1:
                self.node_prev_nodes[node_id].add(cur_nid)
                cur_nid = self.node_prev_node_map[cur_nid][0]

            cur_nid = node_id
            # 后继
            while len(self.node_next_node_map[cur_nid]) == 1:
                self.node_prev_nodes[cur_nid].add(node_id)
                cur_nid = self.node_next_node_map[cur_nid][0]
                
        exclude_keypoints = set()
        # 找出入度不一致的点,加入关键点
        for k in coord_node_map:
            key = coord_node_map[k]
            out_cnt = len(node_out_map[key])
            in_cnt = len(node_in_map[key])
            if out_cnt != in_cnt:
                idx = len(self.keypoints)
                pt = self.nodes[key]['geometry']
                self.keypoints[idx] = (pt.x, pt.y)
                exclude_keypoints.add(str(pt.x) + ',' + str(pt.y))

        # 初始化关键点
        for _, da in das.iterrows():
            div_id = int(da['DIVIDER_ID'])
            if div_id not in self.dividers:
                continue

            divider = self.dividers[div_id]
            pt = divider.geometry.coords[0]
            k = str(pt[0]) + ',' + str(pt[1])
            if k in exclude_keypoints:
                continue

            idx = len(self.keypoints)
            self.keypoints[idx] = pt

        track = [132, 133, 122, 114, 106, 98, 90, 82, 46]
        track.reverse()
        self._init_data_(track)

    def _load_v4_(self):
        dividers = gpd.read_file('easy1/HD_DIVIDER.shp').set_crs('epsg:4326').to_crs(21480)
        das = gpd.read_file('easy1/HD_DIVIDER_ATTRIBUTE.dbf').set_crs('epsg:4326').to_crs(21480)
        div_nodes = gpd.read_file('easy1/HD_DIVIDER_NODE.shp').set_crs('epsg:4326').to_crs(21480)
        # 转坐标系

    def _clean_(self):
        # sample测试
        self.it = 0
        self.loop = 0
        # 前一组打断点
        self.prev_actions = None
        # 保存优化后最终生成的打断线
        self.opt_break_lines = list()
        # 保存首次生成的打断线
        self.org_break_lines = list()
        self.org_break_lines_str = list()
        # 保存参数化的打断线信息
        self.param_break = list()
        self.add_breakline = False
        # 保存打断点关联的打断线
        self.keypoint_breakline_map = dict()
        # 保存div_node跟da的映射关系
        # 轨迹线
        self.track_index = 0
        # 累计得分
        self.score = 0
        # 保存的车道组
        self.lane_group = dict()
        # 按组编在一起的打断点
        self.break_points = dict()
        # 已消耗的关键点
        self.consume_keypoints = dict()
        # 打断点前趋线段(用于排除state)
        self.exclude_nodes = set()
        # 累计打断线总长度
        self.break_length = 0
        # 保存车道线拥有的打断点
        self.divider_breaks = dict()
        # 保存未打断部分的主辅线总长度
        self.rest_len = self.total_len
        # 车道组增量
        self.delta_group = 0
        # 
        self.done = False
        self.info = ''

    def _init_data_(self, track_ids):
        # 初始化轨迹点(用固定divider)
        for _, track_id in enumerate(track_ids):
            divider = self.dividers[track_id]
            if _ == 0:
                self.track.append(divider.geometry.coords[0])

            for i, coord in enumerate(divider.geometry.coords):
                if i == 0:
                    continue

                self.track.append(coord)
            
        for k in self.keypoints:
            # 构建形点
            pt = Point(self.keypoints[k])
            self.keypoints_geo.append(pt)
        self.keypoint_tree = STRtree(self.keypoints_geo)
        # 初始化形点线段/四叉树/拓扑关系
        self._gen_segment_(self.nodes)

    def step(self, action):
        '''
        action为一组打断点,形如:
        [
            [start_id, end_id, t],  // 起点id,终点id,起终点之间的参数t
            ...
        ]
        返回:
        state state包含当前位置半径L内的相关数据,由两部分组成:
            1. 状态矩阵: 状态矩阵有两层，第一层为线段属性, 第二层为对应线段上的打断参数t(t>=0 && t < 1),未打断时为1
            2. 线段的邻接矩阵
        每step一次移动一个形点,直到轨迹线末端结束,然后挑选一个新的轨迹继续移动,直到所有轨迹走完
        1.能够构成车道组的打断线都有正奖励,但是不一定经过关键点,而空打断的奖励为0,会导致每一步都做打断
          为了避免这种情况,令不经过关键点的打断线奖励为-1
        '''
        reward = 0
            
        if len(self.consume_keypoints) == len(self.keypoints):
            self.done = True
            self.info = 'not keypoint to break'
            self._clean_batch_()
            return None, -1, self.done, self.done, {'info': self.info}
        
        if len(self.track) <= self.track_index:
            self.done = True
            rest_keypoint = len(self.keypoints) - len(self.consume_keypoints)
            if rest_keypoint != 0:
                reward += -10 * rest_keypoint
            self.info = 'rest keypoint: ' + str(rest_keypoint)
            self._clean_batch_()
            reward += -1
            return self.cur_state, reward, self.done, self.done, {'info': self.info}

        self._move_()
        reward = self._score_(action)

        if len(self.consume_keypoints) == len(self.keypoints):
            self.done = True
            self.info = 'success'
            reward += 100
        
        if self.add_breakline:
            self.param_break[-1].append(reward)
        self.add_breakline = False
        # 
        return self.cur_state, reward, self.done, self.done, {'info': self.info}
    
    def reset(self, seed = 3, options = {}):
        '''
        state为轨迹点周围可视范围内的divider形点构建的图,包括形点属性和邻接矩阵
        '''
        res = self._reset_(self.name, seed)
        return res
        
    def _reset_(self, name, seed):
        self._clean_()
        self.cur_pos = self.track[self.track_index]
        # self.cur_state = {'node': None, 'matrix': None}
        self.cur_state = self._get_state_()
        return self.cur_state

    def render(self):
        if self.clock is None:
            self.clock = pygame.time.Clock()

        scale = 20
        border = 10
        trans_x = int((self.frame_size[0] + self.L + border) / 2)
        trans_y = int((self.frame_size[1] + self.L + border) / 2)
        cur_x = trans_x + self.cur_pos[0]
        cur_y = trans_y + self.cur_pos[1]
        # print('size:', self.frame_size, cur_x, cur_y, trans_x, trans_y)
        
        total_w = int(self.frame_size[0] + 2 * self.L + 2 * border) * scale
        total_h = int(self.frame_size[1] + 2 * self.L + 2 * border) * scale
        # print('wh', total_w, total_h)
        image = Image.new('RGB', (total_w, total_h), 'white')
        draw = ImageDraw.Draw(image)

        for key in self.keypoints:
            keypoint = self.keypoints[key]
            cx = (keypoint[0] + trans_x) * scale
            cy = (keypoint[1] + trans_y) * scale
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill='black')
            # gfxdraw.circle(self.surf, int(cx), int(cy), 2, (2, 2, 2))

        # print(trans_x, trans_y)
        for div_id in self.dividers:
            div = self.dividers[div_id]
            geos = [((g[0] + trans_x)* scale, (g[1] + trans_y) * scale) for g in div['geometry'].coords]
            draw.line(geos, fill='blue')
            # gfxdraw.line(self.surf, int(geos[0][0]), int(geos[0][1]), int(geos[1][0]), int(geos[1][1]), (2, 2, 200))

        for seg_coords in self.org_break_lines:
            geos = [((g[0] + trans_x)* scale, (g[1] + trans_y) * scale) for g in seg_coords]
            draw.line(geos, fill='red')
        # 转换图像为RGB数组
        left, upper = cur_x - self.L, cur_y- self.L
        right, lower = cur_x + self.L, cur_y + self.L

        # left, upper, right, and lower 像素坐标
        viewport = (left * scale, upper * scale, right * scale, lower * scale)
        # font = ImageFont.truetype("arial.ttf", 15)
        font = draw.getfont()
        draw.text((left * scale, upper * scale), 'score', fill='red', font=font)
        # image.save('./debug.png')
        # print('viewport', viewport)
        rgb_image = image.crop(viewport)
        flipped_image = rgb_image.transpose(Image.FLIP_TOP_BOTTOM)
        # rgb_image.save('./viewport.png')

        image_data = flipped_image.tobytes()
        size = flipped_image.size

        if self.screen is None:
            pygame.init()
            self.screen_width = int(viewport[2] - viewport[0])
            self.screen_height = int(viewport[3] - viewport[1])

            if self.render_model == 'human':
                pygame.display.init()
                self.screen = pygame.display.set_mode(size)
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        pygame_image = pygame.image.fromstring(image_data, size, "RGB")
        self.screen.blit(pygame_image, (0, 0))
        if self.render_model == 'human':
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
    def _score_(self, action):
        '''
        奖励分数从-1到+1,且要求action给出的打断动作是顺序连接的
        1. 根据打断点计算奖励,括号为优先级,如果优先级高的两个值相等,比较下一个优先级: 
            打断的车道组总数量(1) + 打断线总长度(2) 
        2. 打断线允许相交,但是两端点距离差不能超过0.5m
        3. 车道组内不允许出现逆向divider和barrier
        4. 不允许没有打断线的孤立打断点存在
        (车道组数量越少越好>打断线总长度越小越好>车道组外的车道线越短越好)
        '''
        act_len = len(action)
        if act_len == 0 or (int(action[0][0]) == 0 and int(action[0][1]) == 0):
            # self.param_break.append([(0, 0, 0)])
            return 0.0
        if act_len == 1:
            return -1

        break_coords = []
        param_break = []
        node_keys = self.nodes.keys()
        for start_node_id, end_node_id, t in action:
            # 动作操作的node不存在
            sn_id = int(start_node_id)
            en_id = int(end_node_id)
            if sn_id not in node_keys or en_id not in node_keys:
                return -1
            # 参数错误
            if t < 0 or t > 1:
                return -1

            start_node = self.nodes[sn_id]
            end_node = self.nodes[en_id]
            param_break.append((start_node, end_node, t))
            break_coord = self._get_coord_by_t_(start_node, end_node, t)
            break_coords.append(break_coord)
        
        # 检查首尾是否在同一个divider上
        if self._is_connective_(int(break_coords[0][0]), 0, action[1]) or self._is_connective_(int(break_coords[-1][0]), 0, action[0]):
            self.done = True
            return -1

        self.org_break_lines.append(break_coords)
        self.org_break_lines_str.append(LineString(break_coords))

        # for pb in param_break:
        #     bli = BreakLineInfo(pb)
        #     self.param_break.append
        self.param_break.append([pb for pb in param_break])
        self.add_breakline = True
        # 计算打断线长度
        cur_len = 0
        for i, coord in enumerate(break_coords[1:], 1):
            pre_coord = break_coords[i - 1]
            c_x = coord[0]
            c_y = coord[1]
            prev_x = pre_coord[0]
            prev_y = pre_coord[1]
            cur_len += self._distance_(c_x, c_y, prev_x, prev_y)
        if cur_len == 0:
            self.info = 'break len is 0'
            self.done = True
            return -1
        # 两端的车道线方向不一致
        first_start_id = int(action[0][0])
        first_start_coord = self.nodes[first_start_id]
        last_start_id = int(action[-1][0])
        last_start_coord = self.nodes[last_start_id]
        org_break_line = [(break_coords[0][0], break_coords[0][1]), (break_coords[-1][0], break_coords[-1][1])]
        first_orient = self._orient_(org_break_line[0], org_break_line[1], first_start_coord)
        last_orient = self._orient_(org_break_line[0], org_break_line[1], last_start_coord)
        dir = first_orient * last_orient
        if dir < 0:
            self.done = True
            self.info = 'break in reverse directions'
            return -1
        elif dir == 0:
            # print('node', first_start_id, 'or', last_start_id, "in break line")
            pass

        # 排除前趋线段
        if first_start_id in self.exclude_nodes or last_start_id in self.exclude_nodes:
            self.done = True
            return -1

        self._add_prev_nodes(self.org_break_lines_str[-1])

        prev_consume_point = len(self.consume_keypoints)
        if len(self.break_points) == 0:
            # 首次打断,无法构建车道组
            self.break_points[0] = break_coords
            break_line = self.org_break_lines_str[-1]
            keypoint_indexes = self.keypoint_tree.query(break_line.buffer(0.5))
            for pt_index in keypoint_indexes:
                keypoint = self.keypoints_geo[pt_index]
                if self._distance_to_breakline_(keypoint, action) > self.max_distance:
                    continue
                
                self.consume_keypoints[pt_index] = keypoint

            self.prev_actions = action
            c = len(self.consume_keypoints) - prev_consume_point
            if c == 0:
                self.info = 'first time no key point is consume'
                self.done = True
                return -1
            score = c/(cur_len * (1 + c))
            self.info = 'first key point is consume'
            return score
        # TODO: 打断线都在同一个车道线的场景
        # 尝试构建车道组
        if self._generate_lane_group_(action) is False:
            # print('len:', -cur_len)
            self.done = True
            self._clean_batch_()
            return -1

        prev_group_cnt = len(self.lane_group)
        if prev_group_cnt == 0:
            self.done = True
            self.info = 'prev lane group is 0'
            self._clean_batch_()
            return -1

        if prev_group_cnt > 1:
            last_lg_id = prev_group_cnt
            prev_lane_group = self.lane_group[last_lg_id - 1]
            cur_lane_group = self.lane_group[last_lg_id]
            # print('prev lg:', prev_lane_group, 'cur lg:', cur_lane_group)
            # 检查车道组是否与上一车道组相交
            if not cur_lane_group.intersects(prev_lane_group):
                self.done = True
                self.info = 'prev lane group is not intersect'
                self._clean_batch_()
                return -1

            inter = cur_lane_group.intersection(prev_lane_group)
            if inter.area > 0.01:
                self.info = 'lane group intersection area is ' + str(inter.area)
                self.done = True
                self._clean_batch_()
                return -1

        self.break_length += cur_len
        # TODO: 如果打断到末尾,计算一次不被cover的车道线长度,然后清空之前的打断信息,做一次统计
        if self._is_break_tail_(action):
            self._clean_batch_()
        else:
            self.prev_actions = action
        
        c = len(self.consume_keypoints) - prev_consume_point
        # self.delta_group = len(self.lane_group) - prev_group_cnt
        # score = cur_len * (1 + 0.1 * self.delta_group) + self.rest_len
        # score = 1/(cur_len * (1 + 1/c) + self.rest_len/len(self.org_break_lines))
        N = len(self.org_break_lines)
        # TODO: 尽量让打断线经过关键点,所以这个距离也要算入得分里头
        score = N * cur_len/(N * cur_len * (1 + c) + self.rest_len * c)

        self.score += score
        # print('score:', score, 'len:', cur_len, 'total:', self.break_length, 'delta:', delta_group)
        return score

    def _is_break_tail_(self, actions):
        # TODO: 末尾点到打断线的距离小于0.5
        return False

    def _all_in_exclude_nodes_(self, nodes):
        for id in nodes:
            if id not in self.exclude_nodes:
                return False
        return True

    def _add_prev_nodes(self, break_line):
        # 取打断线相交点的前趋nodes
        inters = self._query_intersections_(break_line)
        for inter in inters:
            node_id = int(inter[0])
            if node_id in self.exclude_nodes:
                continue

            self.exclude_nodes.add(node_id)
            prev_nodes = list(self.node_prev_nodes[node_id])
            while len(prev_nodes) != 0:
                front_id = prev_nodes.pop(0)
                if front_id in self.exclude_nodes:
                    continue
                self.exclude_nodes.add(front_id)
                prev_nodes.append(front_id)

    def _clean_batch_(self):
        # TODO: 清空打断信息
        cover_len = 0
        cal_cnt = 0
        if len(self.lane_group) == 0:
            return
        for lg_id in self.lane_group:
            # 统计剩余未打断总长度
            poly_lg = self.lane_group[lg_id]
            # 计算完全圧盖的线段
            segm_indexes = self.segm_tree.query(poly_lg)
            for seg_idx in segm_indexes:
                seg_linestr = self.segments_str[seg_idx + 1]
                if poly_lg.intersects(seg_linestr) is False:
                    continue

                if poly_lg.covers(seg_linestr):
                    cover_len += seg_linestr.length
            cal_cnt += 1
            if cal_cnt == self.delta_group:
                break
        self.rest_len -= cover_len

    def _move_(self):
        if self.done:
            self.info = 'not move'
            return

        self.track_index += 1
        if self.track_index > len(self.track) - 1:
            return

        self.cur_pos = self.track[self.track_index]

        # 收集半径L范围内点数据和拓扑关系
        self.cur_state = self._get_state_()

    # def _rest_la
    def _is_connective_(self, node_id, t, prev):
        end_id = int(prev[0])
        if node_id == end_id and t >= prev[2]:
            # 说明连通
            return True
        else:
            # 往前找node
            prev_nodes = self.node_prev_nodes[node_id]
            if end_id in prev_nodes:
                return True
            return False
    
    def _query_intersections_(self, linestr):
        # 返回相交的线段的起点node_id
        inters_info = []
        buffer = linestr.buffer(0.01)
        segm_indexes = self.segm_tree.query(buffer)
        for seg_idx in segm_indexes:
            seg_id = seg_idx + 1
            seg = self.segments[seg_id]
            seg_linestr = self.segments_str[seg_id]
            if buffer.distance(seg_linestr) > 0.001:
                continue

            inters = linestr.intersection(seg_linestr)
            t = 0
            start_geo = seg[10:13]
            if not inters.is_empty:
                end_geo = seg[13:16]
                dx = end_geo[0] - start_geo[0]
                t_x = (inters.x - start_geo[0])/dx
                t = max(min(t_x, 1.0), 0.0)
                if t < 0.00001: t = 0
                if t > 0.99999: t = 1
            else:
                start_pt = Point(start_geo)
                d = start_pt.distance(linestr)
                if d > 0.001:
                    t = 1.0
            # print('inter:', t)
            inter = (int(seg[0]), int(seg[1]), t)
            inters_info.append(inter)
        return inters_info

    def _is_breakline_connective_(self, left_action, right_action):
        '''
        检查跟打断线相交的divider是否跟之前的打断点连通:
            打断线交点所在divider前趋是否跟前一根打断线相交,且相交divider数量大于等于2
        '''
        prev_left = self.prev_actions[0]
        prev_right = self.prev_actions[-1]
        left_node_id = int(left_action[0])
        left_t = left_action[2]
        right_node_id = int(right_action[0])
        right_t = right_action[2]
        conn_count = 0
        exclude_ids = set()
        if self._is_connective_(left_node_id, left_t, prev_left):
            conn_count += 1
            exclude_ids.add(left_node_id)
        if self._is_connective_(right_node_id, right_t, prev_right):
            conn_count += 1
            exclude_ids.add(right_node_id)
        if conn_count == 2:
            return True

        # 找前一根打断线跟divider的交点
        prev_linestr = self.org_break_lines_str[-2]
        prev_inters = self._query_intersections_(prev_linestr)

        linestr = self.org_break_lines_str[-1]
        segm_indexes = self.segm_tree.query(linestr)
        for seg_idx in segm_indexes:
            # check intersection
            seg_id = seg_idx + 1
            seg = self.segments[seg_id]
            if seg[0] in exclude_ids:
                continue

            seg_linestr = self.segments_str[seg_id]
            if linestr.intersects(seg_linestr) is False:
                continue

            for pre_inter in prev_inters:
                if self._is_connective_(seg[0], 0, pre_inter):
                    conn_count += 1
                    if conn_count >= 2:
                        return True
        return False

    def _generate_lane_group_(self, action):
        '''
        检测打断后车道组是否合法(不合法则失败结束):
        1. 检查打断线两头端点的divider是否同向,失败给-1分
        2. 将打断点串起来,然后将打断线末尾的点沿divider后退方向串起来,检查是否能形成闭环,失败给-1分 (保存新的打断点及车道组面)
        3. 检查闭环内的divider,存在反向divider给-1分
        4. 检查闭环内的divider,存在导流带/道路边缘等给-1分
        5. 检查打断线跟最近关键点距离是否小于0.5, 大于给-1分
        '''
        left_action, right_action = action[0], action[-1]
        prev_left = self.prev_actions[0]
        prev_right = self.prev_actions[-1]
        # 检查跟上一次的打断点是否连通,
        if not self._is_breakline_connective_(left_action, right_action):
            # TODO: 如果没有连通,可能进入到一段新的道路
            self.info = 'generate lane group fail(break line not connected)'
            return False

        # print('connective', prev_left, prev_right)
        polyline = []
        action_list = action.numpy()
        for act in action_list[::-1]:
            start_id = int(act[0])
            end_id = int(act[1])
            coord = self._get_coord_by_t_(self.nodes[start_id], self.nodes[end_id], act[2])
            polyline.append(coord)
        
        left_line = self._get_connectivity_(int(left_action[0]), int(left_action[1]), left_action[2], prev_left)
        if len(left_line) == 0:
            print('empty left side')
        polyline += left_line

        for act in self.prev_actions:
            start_id = int(act[0])
            end_id = int(act[1])
            coord = self._get_coord_by_t_(self.nodes[start_id], self.nodes[end_id], act[2])
            polyline.append(coord)

        reverse_right = self._get_connectivity_(int(right_action[0]), int(right_action[1]), right_action[2], prev_right)
        if len(reverse_right) == 0:
            print('empty right side')
        for coord in reverse_right[::-1]:
            polyline.append(coord)

        max_id = len(self.lane_group) + 1
        try:
            poly_group = Polygon(polyline)
            if not poly_group.is_valid:
                # print('valid lanegroup')
                self.info = 'generate lane group fail(lg invalid)'
                return False

            poly_buffer = poly_group.buffer(0.5)    
            keypoint_indexes = self.keypoint_tree.query(poly_buffer)
            consume_count = len(self.consume_keypoints)
            for pt_index in keypoint_indexes:
                if pt_index in self.consume_keypoints:
                    continue

                keypoint = self.keypoints_geo[pt_index]
                if not poly_buffer.intersects(keypoint):
                    continue

                # 检查闭环内的关键点，距离新打断线是否超过0.5m
                # if self._distance_to_breakline_(keypoint, self.prev_actions) > self.max_distance:
                #     continue
                
                if self._distance_to_breakline_(keypoint, action) > self.max_distance:
                    continue
                
                self.consume_keypoints[pt_index] = keypoint
            if len(self.consume_keypoints) - consume_count == 0:
                self.info = 'no key point is consume'
                return False

            self.lane_group[max_id] = poly_group
            # print('polygon', max_id, len(self.consume_keypoints), ':', poly_group)
            # if max_id == 6:
            #     print('polygon', max_id, ':', poly_group)
            #     print('reverse_right',':', LineString(reverse_right))
        except Exception as e:
            print('except lanegroup:', e)
            self.info = 'generate lane group fail(except)'
            return False
        return True

    def _distance_to_breakline_(self, point, prev_actions):
        break_coords = []
        for action in prev_actions:
            start_id = int(action[0])
            end_id = int(action[1])
            coord = self._get_coord_by_t_(self.nodes[start_id], self.nodes[end_id], action[2])
            break_coords.append(coord)
        break_line = LineString(break_coords)
        d = point.distance(break_line)
        # print('distance:', d, 'point:', point, 'line:', break_line)
        return d

    def _get_connectivity_(self, start_id, end_id, t, prev):
        conn_coords = []
        start_node = self.nodes[start_id]
        end_node = self.nodes[end_id]
        conn_coords.append(self._get_coord_by_t_(start_node, end_node, t))
        if start_id != prev[0] or t < prev[2]:
            prev_keys = list(self.node_prev_node_map.keys())
            prev_id = start_id
            while prev_id != prev[0]:
                if prev_id in prev_keys:
                    prev_ids = self.node_prev_node_map[prev_id]
                    if len(prev_ids) != 1: # 分离合并或者断头场景
                        break

                    # print(prev_id, prev[0])
                    start_node = self.nodes[prev_ids[0]]
                    end_node = self.nodes[prev_id]
                    conn_coords.append(self._get_coord_by_t_(start_node, end_node, 1.))
                    prev_id = prev_ids[0]
                else:
                    break
        return conn_coords

    def _orient_(self, start, end, check):
        start_x, start_y = start
        end_x, end_y = end
        check_coord = check['geometry']
        check_x = check_coord.x
        check_y = check_coord.y
        # v>0，说明点在线的左边，小于在右边，等于则在线上
        return (end_x - start_x) * (check_y - start_y) - (end_y - start_y) * (check_x -start_x)
        
    def _gen_segment_(self, valid_nodes):
        segments = []
        keys = valid_nodes.keys()
        # 构建线段,记录拓扑关系
        for node_id in valid_nodes:
            if node_id not in self.node_next_node_map:
                continue

            next_node_ids = self.node_next_node_map[node_id]
            start_geo = valid_nodes[node_id]['geometry']
            for next_node_id in next_node_ids:
                if next_node_id not in keys:
                    continue

                inter_div = self.node_div_map[node_id] & self.node_div_map[next_node_id]
                div_id = inter_div.pop()
                da = self.das[div_id]
                end_geo = valid_nodes[next_node_id]['geometry']
                # start_node_id, end_node_id, main_aux, virtual, color, type, group type, material, width, overlay, start_x, start_y, start_z, end_x, end_y, end_z
                seg = [node_id, next_node_id, float(da['MAIN_AUX']), float(da['VIRTUAL']), float(da['COLOR']), float(da['TYPE']), float(da['GROUP_TYPE']),
                    float(da['MATERIAL']), float(da['WIDTH']), float(da['OVERLAY']),
                    float(start_geo.x), float(start_geo.y), float(start_geo.z),
                    float(end_geo.x), float(end_geo.y), float(end_geo.z)
                    ]
                seg_linestr = LineString([(float(start_geo.x), float(start_geo.y)), (float(end_geo.x), float(end_geo.y))])
                segments.append(seg_linestr)

                seg_index = len(segments)
                self.seg_next_node_map[seg_index].add(next_node_id)
                self.node_next_seg_map[node_id].add(seg_index)
                self.segments[seg_index] = seg
                self.segments_str[seg_index] = seg_linestr

        self.segm_tree = STRtree(segments)
        return segments

    def _gen_adj_index_(self, segments):
        seg_len = len(segments)
        adj_index = []
        set_list = list(segments)
        for seg_i in segments:
            row = np.zeros(seg_len)
            if seg_i != 0:
                node_set = self.seg_next_node_map[seg_i]
                for node_id in node_set:
                    next_seg_set = self.node_next_seg_map[node_id]
                    for next_seg_id in next_seg_set:
                        if next_seg_id in segments:
                            idx = set_list.index(next_seg_id)
                            row[idx] = 1
            adj_index.append(row)
        # 转置+对角矩阵
        adj = np.array(adj_index)
        adj = adj + np.transpose(adj) + np.eye(len(segments), dtype = int)
        return adj

    def _get_segments_and_adj(self, nodes):
        segment_set = set()
        for node_id in nodes:
            if node_id not in self.node_next_node_map:
                continue

            next_segments = self.node_next_seg_map[node_id]
            for seg_idx in next_segments:
                segment_set.add(seg_idx)

        # 补充空节点,对应空动作
        segment_set.add(0)

        segs = []
        for seg_idx in segment_set:
            if seg_idx == 0:
                segs.append([0 for _ in range(16)])
            else:
                segs.append(self.segments[seg_idx])
        adj_index = self._gen_adj_index_(segment_set)
        
        return segs, adj_index

    def _get_state_(self):
        pt = Point(self.cur_pos[0], self.cur_pos[1])
        pt_buffer = pt.buffer(self.L)
        seg_indexes = self.segm_tree.query(pt_buffer)
        valid_nodes = set()
        for idx in seg_indexes:
            seg = self.segments[idx + 1]
            start_node_id = int(seg[0])
            end_node_id = int(seg[1])
            if start_node_id in self.exclude_nodes or end_node_id in self.exclude_nodes:
                continue

            valid_nodes.add(start_node_id)
            valid_nodes.add(end_node_id)

        # print('new:', len(nodes))
        segments, adj_index = self._get_segments_and_adj(valid_nodes)
        return np.array(segments), adj_index

    def _distance_(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)

    # def _distance_v_(self, points):
    #     dx = []
    #     dy = []
    #     cnt = len(points) - 1
    #     for i in range(cnt):


    def _get_coord_by_t_(self, start_node, end_node, t):
        start_x = start_node['geometry'].x
        end_x = end_node['geometry'].x
        x = start_x + t * (end_x - start_x)
        start_y = start_node['geometry'].y
        end_y = end_node['geometry'].y
        y = start_y + t * (end_y - start_y)
        return (x, y)

    def _distance3_(self, start_coord, end_coord):
        pass
    
    def _find_prev_break_point_(self, start_coord, end_coord, t):
        # 找当前点的前一个打断点
        pass

    def _total_div_len_(self):
        total_l = 0
        for div in self.dividers:
            geos = div['geometry']
            for i, coord in enumerate(geos, 1):
                pre_coord = geos[i - 1]
                total_l += self._distance_(coord[0], coord[1], pre_coord[0], pre_coord[1])
        return total_l
    
    def close(self):
        pygame.quit()

    def save(self, path):
        '''
        保存打断线坐标
        '''
        sid = []
        eid = []
        ts = []
        lines = []
        rewards = []
        U = []
        if len(self.param_break) == 0:
            print('not break success')
            return
        for segms in self.param_break:
            str_sid = ''
            str_eid = ''
            str_t = ''
            coords = []
            for idx in range(len(segms) - 1):
                if len(segms[idx]) == 0:
                    str_sid = '0'
                    str_eid = '0'
                    str_t = '0'
                    break

                start_node = segms[idx][0]
                end_node = segms[idx][1]
                t = segms[idx][2].item()
                start_x = start_node['geometry'].x
                end_x = end_node['geometry'].x
                start_y = start_node['geometry'].y
                end_y = end_node['geometry'].y
                x = start_x + t * (end_x - start_x) + self.ref_coord[0]
                y = start_y + t * (end_y - start_y) + self.ref_coord[1]
                str_sid += start_node['ID'] + ','
                str_eid += end_node['ID'] + ','
                str_t += str(t) + ','
                coords.append(Point(x, y))
            linestr = LineString(coords)
            lines.append(linestr)
            sid.append(str_sid)
            eid.append(str_eid)
            ts.append(str_t)
            reward = segms[-1]
            if len(U) == 0:
                U.append(reward)
            else:
                U.append(U[-1] + reward)
            rewards.append(reward)
        df = pd.DataFrame({'start_node': sid, 'end_node': eid, 't': ts, 'reward': rewards, 'U': U})
        gdf = gpd.GeoDataFrame(df, geometry=lines)
        # print(gdf)
        gdf = gdf.set_crs(21480).to_crs('epsg:4326')
        gdf.to_file(path, driver='GeoJSON')

    def sample(self, version):
        if version == 1:
            return self.sample_v1(3)
        else:
            print("not support sample version")
            pass

    def sample_v1(self, select = 0):
        # select = 3 #random.choice(range(1, 10))
        if select == 0: # 随机
            if random.uniform(0, 1) < 0.5:
                return []
            seg_cnt = len(self.segments)
            start_idx = random.randint(0, seg_cnt)
            end_idx = random.randint(0, seg_cnt)
            t1 = random.uniform(0, 1)
            t2 = random.uniform(0, 1)
            if start_idx not in self.segments or end_idx not in self.segments:
                return []
            start_seg = self.segments[start_idx]
            end_seg = self.segments[end_idx]
            return [[int(start_seg[0]), int(start_seg[1]), t1], [int(end_seg[0]), int(end_seg[1]), t2]]
        elif select == 1:
            return [[2,3, 1.1]]
        elif select == 2:
            return [[0, 0, 0], [0, 0, 0]]
        else:
            # actions = [
            #     [[586, 587, 0.0], [527, 528, 0.0]],
            #     [2],
            #     [[446, 447, 0.0], [540, 696, 0.6]],
            #     [9],
            #     [[713, 466, 0.5], [550, 716, 0.54]],
            #     [1],
            #     [[469, 717, 0.3], [551, 552, 0.8]],
            #     [[469, 717, 0.9999], [552, 720, 0.88]],
            #     [4],
            #     [[725, 476, 0.01], [557, 728, 0.999]],
            #     [5],
            #     [[564, 565, 0.98], [484, 485, 0.9]],
            #     [6],
            #     [[500, 733, 0.999], [570, 568, 0.998]],
            #     [[734, 628, 0.79], [519, 520, 0.97]],
            #     [9],
            #     [[692, 643, 0.27], [578, 693, 0.95]],
            #     [2],
            #     [[646, 647, 1.0], [584, 585, 1.0]],
            #     [2000]
            # ]
            actions = [
                [[540.0,696.0,0.0], [648.0,649.0,0.0]],
                [2],
                [[587.0,588.0,0.0], [533.0,534.0,0.0]],
            ]
            if self.it >= len(actions): self.it = len(actions) - 1
            if len(actions[self.it]) == 1:
                self.loop += 1
                if self.loop == actions[self.it][0]:
                    self.it += 1
                return []
            self.it += 1
            self.loop = 0
            print('step:', self.it)
            return torch.tensor(actions[self.it - 1], dtype = torch.float32)
            
def basic_main(env):
    state = env.reset()
    U = 0
    step = 0
    # print(state[0][9])
    # print(state[1][9])
    while True:
        # action = env.sample(step)
        action = env.sample(1)
        # print(action)
        state, reward, done, truncated, info = env.step(action)
        # env.render()
        # print('reward', reward, '\tinfo', info['info'])
        print('reward', reward, '\tlen', len(state[0]))
        # print(state[0])
        U += reward
        if done is True:
            env.save('./break_info.geojson')
            print('U: ', U)
            env.reset()
            step += 1
            if step > 0:
                break
            U = 0
            continue
        # print(U, action, reward)

if __name__ == "__main__":
    env = BreakEnv('Break-v1', 'human')
    basic_main(env)