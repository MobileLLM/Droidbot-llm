import logging
import collections
import copy
import logging
import random
import time
import math
import os
import requests
import json
import re
import yaml

import numpy as np
import pandas as pd

from .input_event import *
from .input_policy import UtgBasedInputPolicy
from .input_policy3 import DUMP_MEMORY_NUM_STEPS, MAX_NAV_STEPS, MAX_NUM_STEPS_OUTSIDE, MAX_START_APP_RETRY, RANDOM_EXPLORE_PROB, Utils, Memory

'''below is for manual mode'''
ADDTEXT = True
SELECTMODE = False
Manual_mode = os.environ['manual'] == 'True'

GOBACK_element = {
                'allowed_actions': ['press'],
                'status':[],
                'desc': '<button bound_box=0,0,0,0>go back</button>',
                'event_type': 'press',
                'bound_box': '0,0,0,0',
                'class': 'android.widget.ImageView',
                'content_free_signature': 'android.widget.ImageView',
                'size': 0,
                'semantic_element_title': '<button bound_box=0,0,0,0>go back</button>'
            }
RESTART_element = {
                'allowed_actions': ['restart'],
                'status':[],
                'desc': '<button bound_box=0,0,0,0>restart</button>',
                'event_type': 'restart',
                'bound_box': '0,0,0,0',
                'class': 'android.widget.ImageView',
                'content_free_signature': 'android.widget.ImageView',
                'size': 0,
                'semantic_element_title': '<button bound_box=0,0,0,0>restart</button>'
            }
WAIT_element = {
                'allowed_actions': ['wait'],
                'status':[],
                'desc': '<button bound_box=0,0,0,0>wait</button>',
                'event_type': 'wait',
                'bound_box': '0,0,0,0',
                'class': 'android.widget.ImageView',
                'content_free_signature': 'android.widget.ImageView',
                'size': 0,
                'semantic_element_title': '<button bound_box=0,0,0,0>wait</button>'
            }

def _save2yaml(file_name, state_prompt, idx, inputs=None, action_type='touch', state_str=None, structure_str=None, tag=None, width=None, height=None):
    if not os.path.exists(file_name):
        tmp_data = {
        'step_num': 0,
        'records': []
        }
        with open(file_name, 'w', encoding='utf-8') as f:
            yaml.dump(tmp_data, f)

    with open(file_name, 'r', encoding='utf-8') as f:
        old_yaml_data = yaml.safe_load(f)
    
    new_records = old_yaml_data['records']
    new_records.append(
            {'State': state_prompt,
            'Choice': idx,
            'Action': action_type,
            'Input': inputs,
            'state_str': state_str,
            'structure_str': structure_str,
            'tag':tag,
            'width':width,
            'height':height}
        )
    data = {
        'step_num': len(list(old_yaml_data['records'])),
        'records': new_records
    }
    with open(file_name, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)

class Mixed_Guided_Policy(UtgBasedInputPolicy):
    '''
    DFS strategy to deal with known states, if in the end, try to find `back, cancel`, if not work to use restart 
    When new state coming, manually select
    '''
    def __init__(self, device, app, random_input):
        super(Mixed_Guided_Policy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.memory = Memory(utg=self.utg, app=self.app)
        self.previous_actions = []
        self._nav_steps = []
        self._num_steps_outside = 0
        
        # # for manually generating UTG
        # self.manual = Manual_mode
        self._visited = {} # {state_str}, {0, 1}
        self._path2state = {} # {state_str}_{element_id}_{action_id}, {state_str}
        self._masked_ele ={} # {state_str}_{element_id}, {0, 1}
        self.auto_selection = False
        self.last_path = None
        self.start_state = None

    def generate_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        def returned_action(state, action):
            action_desc = Utils.action_desc(action)
            self.logger.info(f'>> executing action in state {state.state_str}: {action_desc}')
            self.previous_actions.append(action)
            return action

        current_state = self.current_state
         
        if self.last_event is not None:
            if self.start_state == None:
                self.start_state = current_state.state_str
            if self._visited.get(current_state.state_str) == None:
                self._visited[current_state.state_str] = 0
                self.auto_selection = False
                print("\033[0;32mNew state\033[0m")
            if self.last_path:
                self._path2state[self.last_path] = current_state.state_str
            executable_action = self._get_action(current_state)
            self.logger.debug("current state: %s" % current_state.state_str)
            return returned_action(current_state, executable_action)
        
        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()
            start_app_action = IntentEvent(intent=start_app_intent)
            self.logger.info("starting app")
            return returned_action(current_state, start_app_action)
        
    def _get_action(self, state):
        element_descs, actiontypes, all_elements = self.parse_all_executable_actions(state)
        element_descs_without_bbox = [re.sub(r'\s*bound_box=\d+,\d+,\d+,\d+', '', desc) for desc in element_descs]
        state_desc = "\n".join(element_descs_without_bbox)
        state_desc_with_bbox = "\n".join(element_descs)
        print('='*80, f'\n{state_desc}\n', '='*80)

        ele_id, act_id, input_text =self.mixed_action_extract(actiontypes, state, element_descs)
        selected_action_type, selected_element = None, None
        if ele_id >= 0:
            selected_action_type, selected_element = actiontypes[ele_id][act_id], all_elements[ele_id]
            if ele_id < len(actiontypes) -3:
                self.last_path = f'{state.state_str}_{ele_id}_{act_id}'

        file_path = os.path.join(self.device.output_dir, 'log.yaml')
        _save2yaml(file_path, state_desc_with_bbox, ele_id, input_text, selected_action_type, state.state_str, state.structure_str, state.tag, state.width, state.height)
        
        if self._visited[self.start_state] == 1:
            print("complete...")
            raise KeyboardInterrupt()
        
        if ele_id == -1:
            print("exit...")
            raise KeyboardInterrupt()
        
        return Utils.pack_action(self.app, selected_action_type, selected_element, input_text)
        
    def auto_action_extract(self, actiontypes, state, elements):
        if self._visited.get(state.state_str) == 1:
            return -1, None
        for i, action in enumerate(actiontypes):
            if re.search(r'(<p.*?>.*?</p>)', elements[i]) or self._masked_ele.get(f"{state.state_str}_{i}"):
                continue
            
            if i == len(actiontypes) - 3:
                break
            for j in range(len(action)):
                _path = f"{state.state_str}_{i}_{j}"
                if  self._visited.get(self._path2state.get(_path)) == 1:
                    continue
                print(f"element id: {i}, action: {actiontypes[i][j]}")
                return i, j

        self._visited[state.state_str] = 1
        print("\033[0;32mCompleted this state\033[0m")
        # todo:: temporary 
        return -1, None
    
    def manual_action_extract(self, actions):
        ele_set, action_set = False, False
        element_id, action_choice = None, None

        size = len(actions)

        while not ele_set:
            try:
                response = input(f"Please input element id:")
                element_id = int(response)
                if element_id >= size and element_id < 0:
                    print('warning, wrong index, please input again')
                    continue
                ele_set = True
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt()
            except:
                print('warning, wrong format, please input again')
                continue
        
        while not action_set:
            try:
                actions_desc = [f'({i}) {actions[element_id][i]}' for i in range(len(actions[element_id]))]
                print('You can choose from: ', '; '.join(actions_desc))
                response = input(f"Please input action id:")
                action_choice = int(response)
                action_set = True
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt()
            except:
                print('warning, wrong format, please input again')
                continue
        
        self.auto_selection = True
        return element_id, action_choice

    def mixed_action_extract(self, actiontypes, state, elements):
        ele_id, act_id = None, None
        if self.auto_selection:
            ele_id, act_id = self.auto_action_extract(actiontypes, state, elements)
            if ele_id == -1:
                ele_id, act_id = self.manual_action_extract(actiontypes)
        else:
            ele_id, act_id = self.manual_action_extract(actiontypes)
        
        if ele_id == -1:
            return -1 ,None, None
        # if self._visited.get(self._path2state.get(f"{state.state_str}_{ele_id}_{act_id}")):
        #     print(f"\033[0;32mThe {state.state_str}_{ele_id}_{act_id} approached state is marked visited\033[0m")
        input_text_value, input_set = None, False 
        if actiontypes[ele_id][act_id] == 'set_text':
            while not input_set:
                try:
                    input_text_value = input(f"Please input the text:")
                    input_set = True
                    break
                except KeyboardInterrupt:
                    raise KeyboardInterrupt()
                except:
                    print('warning, wrong format, please input again')
                    continue
        
        mask_set = False
        id_states = []
        while not mask_set:
            try:
                response = input(f'\033[0;32mPlease input elements list needed to mask, using " " to separate(ele_id is `-1` means set visited):\033[0m')
                if response == '':
                    mask_set = True
                    break
                res = response.strip(" ").split(" ")
                id_states = []
                for x in res:
                    match = re.search(r'^(\d+)-(\d+)$', x)
                    if match:
                        start = int(match.group(1))
                        end = int(match.group(2))
                        for _i in range(start, end+1):
                            id_states.append(_i)
                    else:
                        id_states.append(int(x))
                mask_set = True
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt()
            except:
                print('warning, wrong format, do not record')
                continue
        
        for x in id_states:
            if x == -1:
                self._visited[state.state_str] = 1
                continue
            self._masked_ele[f"{state.state_str}_{x}"] = 1
    
        return ele_id, act_id, input_text_value
        
    def parse_all_executable_actions(self, state):
        state_info = self.memory._memorize_state(state)
        elements = state_info['elements']
        
        element_descs, actiontypes, all_elements = [], [], []  # an element may have different action types, so len(all_elements)>len(elements)

        for element_id, element in enumerate(elements):
            element_desc = f"element {element_id}: {element['full_desc']}"
            all_elements.append(element)
            actiontypes.append(element['allowed_actions'])
            element_descs.append(element_desc)
        state_dict_path = os.path.join(self.device.output_dir, 'StateDicts')
        if not os.path.exists(state_dict_path):
            os.mkdir(state_dict_path)
        state_dict_file = os.path.join(self.device.output_dir, f'StateDicts/{state.tag}.json')
        with open(state_dict_file, 'w') as f:
            json.dump(all_elements, f)
        return element_descs, actiontypes, all_elements
        