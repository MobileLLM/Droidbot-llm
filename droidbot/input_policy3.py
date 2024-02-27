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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
DEBUG = True
ACTION_INEFFECTIVE = 'no effect'
DUMMY_INPUT = 'dummy_user_input'

RANDOM_EXPLORE_PROB = 0.0

MAX_NUM_STEPS_OUTSIDE = 3
MAX_NAV_STEPS = 10
MAX_START_APP_RETRY = 4
ONLY_EXPLORE_IN_APP = True

MAX_NUM_DIFF_ELEMENTS_IN_SIMILAR_STATES = 2
MIN_SIZE_SAME_FUNCTION_ELEMENT_GROUP = 5
SKIP_SIMILAR_ACTION_THRESHOLD = 4
DUMP_MEMORY_NUM_STEPS = 3

EXPLORE_WITH_LLM = False

'''below is for manual mode'''
ADDTEXT = True

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

def _get_GOBACK_element(width, height):
    _GOBACK_element = GOBACK_element.copy()
    
    index = [0, height - 48, width//3, height] # default the GOBACK press is on the left
    index = [str(x) for x in index]

    bound_box = ','.join(index)
    _GOBACK_element['bound_box'] = bound_box
    _GOBACK_element['desc'] = re.sub(r'\s*bound_box=(\d+),(\d+),(\d+),(\d+)', ' bound_box='+bound_box, _GOBACK_element['desc'])
    _GOBACK_element['semantic_element_title'] = re.sub(r'\s*bound_box=(\d+),(\d+),(\d+),(\d+)', ' bound_box='+bound_box, _GOBACK_element['semantic_element_title'])
    return _GOBACK_element

def Get_RESTART_element():
    return RESTART_element

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

# todo:: add the description for element
def _records2yaml(file_name, state_prompt, idx, app_name, selected_states, inputs=None, action_type='touch', state_str=None, structure_str=None, tag=None, width=None, height=None):
    if not os.path.exists(file_name):
        tmp_data = {
            'results' : [
                {
                    'app_name': app_name,
                    'records': [],
                    'selected_elements': [],
                    'selected_records': {},
                    'step_num': 0
                }
            ],
            'comment': None,
            'task_name': None,
        }
        with open(file_name, 'w', encoding='utf-8') as f:
            yaml.dump(tmp_data, f)

    with open(file_name, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)

    res = yaml_data['results'][0] # single app
    record = {'State': state_prompt,
            'Choice': idx,
            'Action': action_type,
            'Input': inputs,
            'state_str': state_str,
            'structure_str': structure_str,
            'tag': tag,
            'width': width,
            'height': height}
    
    index = len(res['records'])
    res['records'].append(record)
    
    if selected_states != None:
        record_copy = record.copy()
        record_copy['State'] = selected_states
        res['selected_elements'].append(record_copy)
        res['selected_records'][index] = record.copy()
        res['step_num'] += 1

    with open(file_name, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f)

'''end for manual mode'''
class GPT:
    def __init__(self):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.history = collections.OrderedDict()

    # gpt-3.5-turbo-1106, gpt-4-1106-preview
    @staticmethod
    def query(prompt, model='gpt-3.5-turbo-1106', url=os.environ['GPT_API_URL'], api_key=os.environ['GPT_API_KEY'], temperature=0.7, verbose=True):
        body = {'model':model, 'messages':[{'role':'user','content':prompt}], 'temperature': temperature}
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}', }
        if verbose:
            print(f'-------- GPT query ---------\n{prompt}')
        if prompt in gpt_inst.history:
            r_content = gpt_inst.history[prompt]
        else:
            response = requests.post(url=url, json=body, headers=headers)
            r = json.loads(response.content)
            r_content = r['choices'][0]['message']['content']
            gpt_inst.prompt_tokens += r['usage']['prompt_tokens']
            gpt_inst.completion_tokens += r['usage']['completion_tokens']
            gpt_inst.history[prompt] = r_content
        if verbose:
            print(f'-------- GPT response ---------\n{r_content}')
        return r_content


gpt_inst = GPT()


class Utils:
    @staticmethod
    def get_action_type(action):
        action_type = action.event_type
        if action_type == KEY_KeyEvent:
            return KEY_KeyEvent
        if action_type == KEY_RestartAppEvent:
            return KEY_RestartAppEvent
        allowed_actions = action.view['allowed_actions']
        status = action.view['status']
        if action_type == KEY_TouchEvent and 'select' in allowed_actions:
            if 'selected' in status:
                return 'unselect'
            else:
                return 'select'
        if isinstance(action, ScrollEvent):
            return f'{action_type} {action.direction}'
        return action_type
    
    @staticmethod
    def pack_action(app, action_type, target_element, input_text):
        action_dict = {'event_type': action_type, 'view': target_element}
        if action_type == KEY_SetTextEvent:
            action_dict['text'] = input_text
        elif 'scroll' in action_type:
            action_dict['event_type'] = KEY_ScrollEvent
            action_dict['direction'] = action_type.split(' ')[-1]
        elif action_type == 'press':
            return KeyEvent(name='BACK')
        elif action_type == "restart":
            return RestartAppEvent(app=app)
        return InputEvent.from_dict(action_dict)
    
    @staticmethod
    def action_desc(action):
        action_type = action.event_type
        desc = action_type
        if action_type in [KEY_IntentEvent]:
            desc += f' {action.intent}'
        if action_type in [KEY_ScrollEvent]:
            desc += f' {action.direction}'
        if action_type in [KEY_KeyEvent]:
            desc += f' {action.name}'
        if action_type in [KEY_TouchEvent, KEY_LongTouchEvent, KEY_SelectEvent, KEY_UnselectEvent, KEY_ScrollEvent, KEY_SetTextEvent]:
            element = action.view
            view_desc = element['desc'] if 'desc' in element else f"<{element['class']}, bound_box={element['bound_box']}>"
            desc += f' {view_desc}'
        if action_type in [KEY_SetTextEvent]:
            desc += f' {action.text}'
        return desc


class Memory:
    def __init__(self, utg, app):
        self.utg = utg
        self.app = app
        self.logger = logging.getLogger(self.__class__.__name__)
        self.known_states = collections.OrderedDict()
        self.semantic_states = collections.OrderedDict()
        self.known_transitions = collections.OrderedDict()
        self.known_structures = collections.OrderedDict()
        self.action_history = pd.DataFrame()
        self.action_effects = pd.DataFrame()
        # GPT.query('hello!', verbose=True) # GPT check
    
    def to_string(self, with_similarity_info=True, with_target_info=True, with_action_effects_info=True):
        memory_str = f'## All pages of app "{self.app.app_name}":\n'
        semantic_states = self.semantic_states
        for si, semantic_state_title in enumerate(semantic_states.keys()):
            state_desc = self.get_semantic_state_desc(semantic_state_title, with_similarity_info, with_target_info)
            memory_str += f'\n{state_desc}'
        if with_action_effects_info:
            memory_str += f'\n\n## Action effects:\n{self.get_action_effects_desc()}\n'
        # print(memory_str)
        return memory_str
    
    def get_action_effects_desc(self, with_element_info=False):
        action_effects_desc = ''
        if len(self.action_effects) == 0:
            return action_effects_desc
        if with_element_info:
            action_effects_desc += self.action_effects.to_string()
        else:
            action_effects_desc += self.action_effects[['from_page', 'to_page', 'action_type', 'elemend_id', 'element_desc', 'text']].to_string()
        return action_effects_desc

    def get_semantic_state_desc(self, semantic_state_title, with_similarity_info=False, with_target_info=True):
        semantic_states = self.semantic_states
        state_desc = f' page {list(semantic_states.keys()).index(semantic_state_title)}: {semantic_state_title}\n'
        semantic_elements = semantic_states[semantic_state_title]['semantic_elements']
        same_function_element_groups = []
        # print(semantic_elements)
        for ei, semantic_element_title in enumerate(semantic_elements.keys()):
            action_targets = semantic_elements[semantic_element_title]['action_targets']
            action_effect_info = []
            # print('action_targets', action_targets)
            if with_target_info:
                for action_type in action_targets:
                    target_state_infos = action_targets[action_type]
                    
                    '''below to add the inputted text into the memory string. You can disable this by setting ADDTEXT = False'''
                    target_state_strs, input_texts = [], []
                    for state_id, target_state_info in enumerate(target_state_infos):
                        input_text = ''
                        if isinstance(target_state_info, list):
                            target_state_strs.append(target_state_info[0])
                            input_text = target_state_info[1]
                        else:
                            target_state_strs.append(target_state_info)
                        input_texts.append(input_text)
                    '''end'''
                    
                    target_semantic_state_titles = self._get_target_semantic_states(target_state_strs)
                    action_effects = []
                    
                    input_text_id = 0
                    for target_semantic_state_title, _ in target_semantic_state_titles:
                        if target_semantic_state_title == ACTION_INEFFECTIVE:
                            action_effects.append(ACTION_INEFFECTIVE)
                            continue
                        # if target_semantic_state_title == semantic_element_title:
                        #     continue
                        target_semantic_state_id = list(semantic_states.keys()).index(target_semantic_state_title)
                        if ADDTEXT and action_type == 'set_text':
                            action_effects.append(f'on set_text(\'{input_texts[input_text_id]}\'), go to page {str(target_semantic_state_id)}')
                        else:
                            action_effects.append(f'go to page {str(target_semantic_state_id)}')
                        input_text_id += 1
                        
                    if not action_effects:
                        continue
                    
                    if ADDTEXT and action_type == 'set_text':
                        action_effect_info.append(", ".join(action_effects))
                    else:
                        action_effect_info.append(f'on {action_type}, {", ".join(action_effects)}')
                    
            if with_similarity_info:
                similar_semantic_elements = semantic_elements[semantic_element_title]['similar_semantic_elements']
                similar_ele_ids = []
                for similar_ele, count in similar_semantic_elements.items():
                    if count > 0:
                        similar_ele_ids.append(list(semantic_elements.keys()).index(similar_ele))
                if len(similar_ele_ids) > 0:
                    same_function_element_group = '{' + ','.join([str(ele_id) for ele_id in sorted(set(similar_ele_ids + [ei]))]) + '}'
                    if same_function_element_group not in same_function_element_groups:
                        same_function_element_groups.append(same_function_element_group)
                    # similar_ele_ids = ','.join([str(ele_id) for ele_id in similar_ele_ids])
                    # action_effect_info.append(f'similar to elements {similar_ele_ids}')
            action_effect_comment = f'// {"; ".join(action_effect_info)}' if action_effect_info else ''
            state_desc += f'  element {ei}: {semantic_element_title} {action_effect_comment}\n'
        if with_similarity_info:
            if len(same_function_element_groups) > 0:
                state_desc += f' same-function elements: {", ".join(same_function_element_groups)}\n'
        return state_desc
    
    def all_states(self, in_app_only=True):
        states = []
        for state_str, state_info in self.known_states.items():
            if in_app_only and state_info['app_foreground_depth'] != 0:
                continue
            states.append(state_info['state'])
        return states

    def _gen_state_semantic_info(self, state, with_llm=EXPLORE_WITH_LLM):
        state_desc, elements = state.text_representation
        if not with_llm:
            state_info = {
                'state': state,
                'activity': state.activity_short_name,
                'app_foreground_depth': state.get_app_activity_depth(self.app),
                'page_description': state.structure_str,
                'elements_description': '',
                'elements': elements,
                'same_function_element_groups': []
            }
            return state_info
        prompt = f'You are a mobile app testing expert. Given a GUI page of an app, ' + \
            'you can precisely understand the main function of the page and each GUI element.\n' + \
            f'Now suppose you are analyzing an app named "{self.app.app_name}", ' + \
            f'the current GUI page shows following elements:\n{state_desc}\n' + \
            'Please think step by step and respond in the following format:\n' + \
            ' Page description: <short (less than 20 words) description of the function of current page>\n' + \
            ' Elements description: <short (less than 20 words) summary of main control elements in current page, comma separated>\n' + \
            ' Same-function elements: <groups of element ids, each group contains multiple elements that lead to the same function ' + \
            '(possibly with different parameters), comma separated. Example: {2,3,4},{7,8} or None.> ' + \
            'The elements with different layouts and redirect targets are less likely to have the same function.\n'
            # 'Should be None in most cases.\n'
            #  0. <short description of element 0>;\n 1. <short description of element 1>;\n ...' + \
            # 'possible use cases: \n <a list of example use cases in this GUI page that may require multiple actions. If the use case involves input text or keyword, provide an example>'
        response = GPT.query(prompt)
        page_description = re.search(r'Page description:(.+)', response)
        elements_description = re.search(r'Elements description:(.+)', response)
        same_function_elements = re.search(r'Same-function elements:(.+)', response)
        page_description = page_description.group(1).strip() if page_description else None
        elements_description = elements_description.group(1).strip() if elements_description else None
        same_function_elements = same_function_elements.group(1).strip() if same_function_elements else None
        same_function_element_groups = []
        if same_function_elements is not None and same_function_elements != 'None':
            matches = re.finditer(r'\{([^}]*)\}', same_function_elements)
            for m in matches:
                element_ids = [int(x) for x in re.findall(r'\d+', m.group(1))]
                if len(element_ids) < MIN_SIZE_SAME_FUNCTION_ELEMENT_GROUP:
                    # there are too few elements in a group, skip
                    continue
                same_function_element_groups.append(set(element_ids))
        state_info = {
            'state': state,
            'activity': state.activity_short_name,
            'app_foreground_depth': state.get_app_activity_depth(self.app),
            'page_description': page_description,
            'elements_description': elements_description,
            'elements': elements,
            'same_function_element_groups': same_function_element_groups
        }
        return state_info
    
    def _classify_state(self, state_info, semantic_states, group_same_structure=True, filter_same_activity=True, filter_similar_elements=True, with_llm=EXPLORE_WITH_LLM):
        state_title = f'{state_info["page_description"]}. Elements: {state_info["elements_description"]}'
        history_states = {}
        history_states_desc = {}
        for i, history_state_title in enumerate(semantic_states.keys()):
            if state_title == history_state_title:
                return history_state_title, state_title
            history_state_info = semantic_states[history_state_title]
            if group_same_structure:
                if state_info['state'].structure_str in history_state_info['states_structures']:
                    return history_state_title, state_title
            if filter_same_activity:
                if state_info['activity'] != history_state_info['activity']:
                    continue
            if filter_similar_elements:
                state_ele_sigs = set([e['content_free_signature'] for e in state_info['elements']])
                history_state_ele_sigs = history_state_info['element_sigs']
                different_ele_sigs = state_ele_sigs.symmetric_difference(history_state_ele_sigs)
                if len(different_ele_sigs) > MAX_NUM_DIFF_ELEMENTS_IN_SIMILAR_STATES:
                    continue
            history_state_id = list(self.semantic_states.keys()).index(history_state_title)
            history_states[history_state_id] = history_state_title
            history_states_desc[history_state_id] = f'page {history_state_id}: {history_state_title}'
            # history_states_desc[i] = self.get_semantic_state_desc(history_state_title, with_similarity_info=False, with_target_info=False)
        if len(history_states) == 0 or not with_llm:
            return None, state_title
        history_states_desc = '\n'.join(history_states_desc.values())
        current_state_desc, _ = state_info['state'].text_representation
        current_state_desc = f'{state_title}\n{current_state_desc}'
        prompt = f'You are a mobile app testing expert. ' + \
            'You can precisely understand the functions of GUI pages and identify the new pages that require additional testing.\n' + \
            f'Now suppose you are analyzing an app named "{self.app.app_name}". There are following previous GUI pages:\n{history_states_desc}\n\n' + \
            f'Given a new page: {current_state_desc}\n' + \
            'Please determine whether this new page is functionally equivalent with a previous page ' + \
            '(e.g. they are the same page with different dynamic content). Respond in the format:\n ' + \
            'Equivalent=<True or False>, if True, also respond "Page id=<id of the equivalent previous page>"'
            #  0. <short description of element 0>;\n 1. <short description of element 1>;\n ...' + \
            # 'possible use cases: \n <a list of example use cases in this GUI page that may require multiple actions. If the use case involves input text or keyword, provide an example>'
        response = GPT.query(prompt)
        state_id = re.search(r'Page id=(\d+)', response)
        state_id = int(state_id.group(1).strip()) if state_id else None
        matched_state_title = None
        if state_id is not None and state_id in history_states:
            matched_state_title = history_states[state_id]
        return matched_state_title, state_title
    
    def _classify_element(self, element, semantic_elements):
        element_title = element['desc']
        if element_title in semantic_elements:
            return element_title, element_title
        # if element_title == ''
        element_tag = re.search(r'<(\S+) ', element_title).group(1)
        element_bound = re.search(r"bound_box=(\d+,\d+,\d+,\d+)", element_title).group(1)
        for element_i_title in semantic_elements:
            element_i_tag = re.search(r'<(\S+) ', element_i_title).group(1)
            element_i_bound = re.search(r"bound_box=(\d+,\d+,\d+,\d+)", element_i_title).group(1)
            if element_i_tag == element_tag and element_i_bound == element_bound:
                return element_i_title, element_title
        # if element_title == '<input bound_box=143,858,1036,974>bob</input>':
        #     import pdb;pdb.set_trace()
        return None, element_title

    def _memorize_state(self, state):
        # if state.get_app_activity_depth(self.app) != 0:
        #     return None
        if state.state_str in self.known_states:
            return self.known_states[state.state_str]
        
        state_info = self._gen_state_semantic_info(state)
        self.known_states[state.state_str] = state_info
        semantic_state_title, state_title = self._classify_state(state_info, self.semantic_states)
        if not semantic_state_title:
            semantic_state_title = state_title
            self.semantic_states[state_title] = {
                'states': [],
                'states_structures': [],
                'semantic_elements': collections.OrderedDict(),
                'activity': state_info['activity'],
                'app_foreground_depth': state_info['app_foreground_depth'],
                'element_sigs': set()
            }
        state_info['semantic_state_title'] = semantic_state_title
        self.semantic_states[semantic_state_title]['states'].append(state.state_str)
        self.semantic_states[semantic_state_title]['states_structures'].append(state.structure_str)

        semantic_elements = self.semantic_states[semantic_state_title]['semantic_elements']
        idx_semantic_element_titles = []
        for i, element in enumerate(state_info['elements']):
            self.semantic_states[semantic_state_title]['element_sigs'].add(element['content_free_signature'])
            semantic_element_title, element_title = self._classify_element(element, semantic_elements)
            # print(element, semantic_element_title)
            if not semantic_element_title:
                semantic_element_title = element_title
                semantic_elements[semantic_element_title] = {'elements': [], 'action_targets': {}, 'similar_semantic_elements': {}}
            element['semantic_element_title'] = semantic_element_title
            semantic_elements[semantic_element_title]['elements'].append((state.state_str, i))
            idx_semantic_element_titles.append(semantic_element_title)
            for action in element['allowed_actions']:
                if action not in semantic_elements[semantic_element_title]['action_targets']:
                    semantic_elements[semantic_element_title]['action_targets'][action] = []
        
        same_function_element_groups = state_info['same_function_element_groups']
        for ele_i, ele_i_title in enumerate(idx_semantic_element_titles):
            for ele_j, ele_j_title in enumerate(idx_semantic_element_titles):
                if ele_i == ele_j:
                    continue
                ele_ij_similar = False
                for ele_group in same_function_element_groups:
                    if ele_i in ele_group and ele_j in ele_group:
                        ele_ij_similar = True
                if ele_j_title not in semantic_elements[ele_i_title]['similar_semantic_elements']:
                    semantic_elements[ele_i_title]['similar_semantic_elements'][ele_j_title] = 0
                semantic_elements[ele_i_title]['similar_semantic_elements'][ele_j_title] += (1 if ele_ij_similar else -1)
        return state_info
    
    def save_transition(self, action, from_state, to_state):
        if not from_state or not to_state:
            return
        action_record = {
            'timestamp': pd.Timestamp.now(),
            'from_state': from_state.state_str,
            'to_state': to_state.state_str,
            'action': Utils.action_desc(action)
        }
        self.action_history = pd.concat([self.action_history, pd.DataFrame([action_record])], ignore_index=True)
        if not isinstance(action, UIEvent):
            if not Manual_mode or isinstance(action, IntentEvent):
                return
            
        from_state_info = self._memorize_state(from_state)
        to_state_info = self._memorize_state(to_state)
        if not Manual_mode and action.view is None:
            return
        action_str = action.get_event_str(state=from_state)
        if action_str in self.known_transitions and self.known_transitions[action_str]['to_state'] == to_state:
            return
        if from_state_info is None:
            return
        
        if isinstance(action, RestartAppEvent):
            element = RESTART_element
        elif isinstance(action, UIEvent):
            element = action.view
        else:
            element = _get_GOBACK_element(to_state.width, to_state.height)
        action_target = ACTION_INEFFECTIVE \
            if from_state.state_str == to_state.state_str \
            else to_state.state_str
        # TODO decide how to represent the effect of an action
        # action_effect = f'{from_state.structure_str}->{action_target}'
        action_effect = action_target
        self.known_transitions[action_str] = {
            'from_state': from_state,
            'to_state': to_state,
            'action': action,
            'action_effect': action_effect
        }
        self.update_action_effects(from_state, to_state, action)  

        from_semantic_state = from_state_info['semantic_state_title']
        to_semantic_state = to_state_info['semantic_state_title']
        semantic_element_title = element['semantic_element_title'] if 'semantic_element_title' in element else element['desc']
        action_targets = self.semantic_states[from_semantic_state]['semantic_elements'][semantic_element_title]['action_targets']
        action_type = Utils.get_action_type(action)
        if action_type not in action_targets:
            self.logger.warn(f'save_transition: action_type {action_type} not available')
        else:
            if ADDTEXT and action_type == 'set_text':
                action_targets[action_type].append([action_target, action.text])
            else:
                action_targets[action_type].append(action_target)

    def update_action_effects(self, from_state, to_state, action):
        if not isinstance(action, UIEvent) and not Manual_mode:  
            return None
        if isinstance(action, RestartAppEvent):
            element = RESTART_element
        elif isinstance(action, UIEvent):
            element = action.view
        else:
            element = _get_GOBACK_element(to_state.width, to_state.height)
        is_effective = from_state.state_str != to_state.state_str
        from_state_title = self.known_states[from_state.state_str]['semantic_state_title']
        from_state_id = list(self.semantic_states.keys()).index(from_state_title)
        to_state_title = self.known_states[to_state.state_str]['semantic_state_title']
        to_state_id = list(self.semantic_states.keys()).index(to_state_title)
        action_type = Utils.get_action_type(action)
        element_desc = element['desc']
        element_status = ','.join(element['status'])
        semantic_element_title = element['semantic_element_title'] if 'semantic_element_title' in element else element['desc']
        element_id = list(self.semantic_states[from_state_title]['semantic_elements'].keys()).index(semantic_element_title)
        element_class = element['class']
        element_size = element['size']
        new_effect = {
            'from_page': from_state_id,
            'to_page': to_state_id,
            'action_type': action_type,
            'elemend_id': element_id,
            'element_desc': element_desc,
            'element_class': element_class,
            'element_size': element_size,
            'element_status': element_status,
            'text': action.text if hasattr(action, 'text') else None,
            'effective': is_effective
        }
        self.action_effects = pd.concat([self.action_effects, pd.DataFrame([new_effect])], ignore_index=True)
        return new_effect
    
    def _get_target_semantic_states(self, target_state_strs):
        semantic_states = []
        for target_state_str in target_state_strs:
            if target_state_str in self.known_states:
                state_info = self.known_states[target_state_str]
                semantic_states.append(state_info['semantic_state_title'])
            elif target_state_str == ACTION_INEFFECTIVE:
                semantic_states.append(target_state_str)
            else:
                self.logger.warn(f'_get_target_semantic_states unknown state_str: {target_state_str}')
        if not semantic_states:
            return []
        semantic_states_ordered = []
        for state, count in collections.Counter(semantic_states).most_common():
            semantic_states_ordered.append((state, count))
        return semantic_states_ordered

    def save_structure(self, state):
        structure_str = state.structure_str
        is_new_structure = False
        if structure_str not in self.known_structures:
            self.known_structures[structure_str] = []
            is_new_structure = True
        self.known_structures[structure_str].append(state)
        return is_new_structure
    
    def get_explored_semantic_actions(self):
        explored_semantic_actions = set()
        for semantic_state_title in self.semantic_states:
            semantic_elements = self.semantic_states[semantic_state_title]['semantic_elements']
            for semantic_element_title in semantic_elements:
                action_targets = semantic_elements[semantic_element_title]['action_targets']
                # similar_semantic_elements = semantic_elements[semantic_element_title]['similar_semantic_elements']
                for action_type in action_targets:
                    target_state_strs = action_targets[action_type]
                    if not target_state_strs:
                        continue
                    explored_semantic_actions.add((semantic_state_title, semantic_element_title, action_type))
                    # also mark the similar elements as explored
                    # for similar_semantic_element in similar_semantic_elements:
                    #     if similar_semantic_elements[similar_semantic_element] > 0:
                    #         explored_semantic_actions.add((semantic_state_title, similar_semantic_element, action_type))
        return explored_semantic_actions

    def get_unexplored_actions(self, find_in_states=[], skip_similar=True, prefer_unique=False):
        unexplored_actions = []
        if not find_in_states:
            return unexplored_actions
        unique_actions = []
        explored_semantic_actions = self.get_explored_semantic_actions()
        for state in find_in_states:
            state_info = self._memorize_state(state)
            semantic_state_title = state_info['semantic_state_title']
            for ei, element in enumerate(state_info['elements']):
                semantic_element_title = element['semantic_element_title']
                # action_targets = semantic_elements[semantic_element_title]['action_targets']
                for action_type in element['allowed_actions']:
                    semantic_action = (semantic_state_title, semantic_element_title, action_type)
                    if semantic_action in explored_semantic_actions:
                        continue
                    from_state_id = list(self.semantic_states.keys()).index(semantic_state_title)
                    element_status = ','.join(element['status'])
                    element_class = element['class']
                    element_size = element['size']
                    element_desc = element['desc']
                    df = self.action_effects
                    if skip_similar and len(self.action_effects) > SKIP_SIMILAR_ACTION_THRESHOLD:
                        # same element across different states
                        df1 = df[(df['element_desc']==element_desc) & (df['element_status']==element_status) & (df['action_type']==action_type)] \
                            [['to_page', 'effective']]
                        if len(df1) > SKIP_SIMILAR_ACTION_THRESHOLD and len(df1.drop_duplicates()) == 1:
                            continue
                        # similar elements in the same state
                        df2 = df[(df['from_page']==from_state_id) & (df['element_class']==element_class) & (df['element_size']==element_size) & \
                            (df['element_status']==element_status) & (df['action_type']==action_type)] \
                            [['to_page', 'effective']]
                        if len(df2) > SKIP_SIMILAR_ACTION_THRESHOLD and len(df2.drop_duplicates()) == 1:
                            continue
                    if prefer_unique and len(self.action_effects) > 1:
                        df3 = df[(df['element_class']==element_class) & (df['element_size']==element_size) & \
                            (df['element_status']==element_status) & (df['action_type']==action_type)] \
                            [['to_page', 'effective']]
                        if len(df3) == 0:
                            unique_actions.append((state, element, action_type))
                    unexplored_actions.append((state, element, action_type))
        if prefer_unique and len(unique_actions) > 0:
            return unique_actions
        return unexplored_actions
    
    def gen_input_text(self, state_desc, target_element, with_llm=EXPLORE_WITH_LLM):
        """
        return a text string that can be the input text for the target element
        """
        if not with_llm:
            return DUMMY_INPUT
        prompt = f'You are a mobile app testing expert. Given a GUI page of an app and an editable text field, ' + \
            'you can generate a meaningful input string for the text field.\n' + \
            f'Now suppose you are analyzing a GUI page with following elements:\n{state_desc}\n' + \
            f'The text field is element id={target_element["local_id"]}. Please respond in the following format:\n' + \
            ' Input text: "<the generated input text>"'
            #  0. <short description of element 0>;\n 1. <short description of element 1>;\n ...' + \
            # 'possible use cases: \n <a list of example use cases in this GUI page that may require multiple actions. If the use case involves input text or keyword, provide an example>'
        response = GPT.query(prompt)
        input_text = re.search(r'Input text: "(.+)"', response)
        input_text = input_text.group(1).strip() if input_text else DUMMY_INPUT
        return input_text
    
    def get_executable_action(self, state=None, element=None, action_type=None, input_text=None):
        if state is None:
            state_str = random.choice(self.known_states.keys())
            state = self.known_states[state_str]['state']
        state_desc, elements = state.text_representation
        if element is None:
            element = random.choice(elements)
        if action_type is None:
            action_type = random.choice(element['allowed_actions'])
        if action_type == KEY_SetTextEvent and input_text is None:
            input_text = self.gen_input_text(state_desc, element) if action_type == KEY_SetTextEvent else None
        return state, Utils.pack_action(self.app, action_type, element, input_text)


class Memory_Guided_Policy(UtgBasedInputPolicy):
    def __init__(self, device, app, random_input):
        super(Memory_Guided_Policy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.memory = Memory(utg=self.utg, app=self.app)
        self.previous_actions = []
        self._nav_steps = []
        self._num_steps_outside = 0
        
        # # for manually generating UTG
        # self.manual = Manual_mode

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
        try:
            self.memory.save_transition(self.last_event, self.last_state, current_state)
        except Exception as e:
            self.logger.warning(f'failed to save transition: {e}')
            import traceback
            traceback.print_exc()
        # self.logger.info(f'we have {len(self.memory.known_transitions)} transitions now')
        
        if Manual_mode and self.last_event is not None:
            executable_action = self.get_manual_action(current_state)
            self.logger.debug("current state: %s" % current_state.state_str)
            self._dump_memory()
            return returned_action(current_state, executable_action)

        if self.last_event is not None:
            self.last_event.log_lines = self.parse_log_lines()
        # interested_apis = self.monitor.get_interested_api()
        # self.monitor.check_env()
        self.logger.debug("current state: %s" % current_state.state_str)
        self._dump_memory()

        nav_action, n_steps = self.navigate(current_state)
        if nav_action:
            self.logger.info(f'navigating, {n_steps} steps left')
            return returned_action(current_state, nav_action)
        self._nav_steps = []  # if navigation fails, stop navigating

        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()
            start_app_action = IntentEvent(intent=start_app_intent)
            self.logger.info("starting app")
            return returned_action(current_state, start_app_action)
        elif current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            self._num_steps_outside += 1
            if self._num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                if self._num_steps_outside > MAX_NUM_STEPS_OUTSIDE + 1:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    start_app_intent = self.app.get_start_intent()
                    go_back_event = IntentEvent(intent=start_app_intent)
                self.logger.info("going back to the app")
                return returned_action(current_state, go_back_event)
        else:
            # If the app is in foreground
            self._num_steps_outside = 0

        steps_since_last_kill = 0
        for previous_action in reversed(self.previous_actions):
            if isinstance(previous_action, KillAppEvent):
                break
            steps_since_last_kill += 1
        if steps_since_last_kill > MAX_NAV_STEPS:
            self.logger.info(f"exploring too long, kill and restart")
            return returned_action(current_state, KillAppEvent(app=self.app))
        
        num_start_app_retry = 0
        for previous_action in reversed(self.previous_actions):
            if isinstance(previous_action, IntentEvent) and previous_action.intent == self.app.get_start_intent():
                num_start_app_retry += 1
            else:
                break
        if num_start_app_retry > MAX_START_APP_RETRY:
            self.logger.info(f"starting app failed for {num_start_app_retry} times, reinstalling the app")
            self.device.uninstall_app(self.app)
            self.device.install_app(self.app)
            self.previous_actions = []
            start_app_intent = self.app.get_start_intent()
            start_app_action = IntentEvent(intent=start_app_intent)
            return returned_action(current_state, start_app_action)

        # # TODO if it is a new structure, try to go back first
        # is_structure_new = self.memory.save_structure(current_state)
        # if is_structure_new:
        #     self.logger.info("it is a new structure, adding go-back transition")
        #     return returned_action(current_state, KeyEvent(name="BACK"))

        if len(self._nav_steps) == 0 and np.random.uniform() > RANDOM_EXPLORE_PROB:
            target_state, target_action = self.pick_target(current_state)
            if target_state:
                # perform target action
                self.logger.info(f"exploring current state")
                return returned_action(current_state, target_action)
            target_state, target_action, nav_steps = self.pick_navigate_target(current_state)
            if target_state:
                # navigate to target action
                self.logger.info(f"exploring state {target_state.state_str}, action: {Utils.action_desc(target_action)}")
                self._nav_steps = nav_steps
                nav_action, n_steps = self.navigate(current_state)
                if nav_action:
                    self.logger.info(f'navigating, {n_steps} steps left')
                    return returned_action(current_state, nav_action)
        self._nav_steps = []  # if navigation fails, stop navigating

        self.logger.info("trying random action")
        # possible_events = current_state.get_possible_input()
        # possible_events.append(KeyEvent(name="BACK"))
        # random.shuffle(possible_events)
        # action = possible_events[0]
        # if isinstance(action, UIEvent) and 'desc' not in action.view:
        #     print('invalid action: ', action.view)
        _, random_action = self.memory.get_executable_action(state=current_state)
        return returned_action(current_state, random_action)

    def pick_target(self, current_state):
        unexplored_actions = self.memory.get_unexplored_actions(find_in_states=[current_state])
        if not unexplored_actions:
            return None, None
        (state, element, action_type) = random.choice(unexplored_actions)
        _, action = self.memory.get_executable_action(state, element, action_type)
        return state, action
    
    def pick_navigate_target(self, current_state, randomly=True, shortest=True):
        unexplored_actions = self.memory.get_unexplored_actions(find_in_states=self.memory.all_states(in_app_only=ONLY_EXPLORE_IN_APP))
        if randomly:
            random.shuffle(unexplored_actions)
        target_state, target_element, target_action_type, nav_steps = None, None, None, None
        for state_, element_, action_type_ in unexplored_actions:
            nav_steps_ = self.get_shortest_nav_steps(current_state, state_)
            if nav_steps_ is None:
                continue
            if nav_steps is None or len(nav_steps_) < len(nav_steps):
                target_state, target_element, target_action_type, nav_steps = state_, element_, action_type_, nav_steps_
                if not shortest:   # no need to return shortest, return now
                    break
        if target_state is None:
            return None, None, None
        _, target_action = self.memory.get_executable_action(target_state, target_element, target_action_type)
        nav_steps = nav_steps + [(target_state, target_action)]
        return target_state, target_action, nav_steps

    def navigate(self, current_state):
        if self._nav_steps and len(self._nav_steps) > 0:
            nav_state, nav_action = self._nav_steps[0]
            self._nav_steps = self._nav_steps[1:]
            nav_action_ = self._get_nav_action(current_state, nav_state, nav_action)
            if nav_action_:
                return nav_action_, len(self._nav_steps)
            else:
                self.logger.warning(f"navigate: failed in state {current_state.state_str}")
                # self.utg.remove_transition(self.last_event, self.last_state, nav_state)  # FIXME how to punish the failed navigation
        return None, 0

    def _get_nav_action(self, current_state, nav_state, nav_action):
        # get the action similar to nav_action in current state
        try:
            # if current_state.structure_str != nav_state.structure_str:
            #     return None
            if not isinstance(nav_action, UIEvent):
                return nav_action
            nav_view = nav_action.view
            nav_view_desc = nav_view['desc']
            new_state_views = current_state.text_representation[-1]
            new_view_idx = [view['desc'] for view in new_state_views].index(nav_view_desc)
            new_view = new_state_views[new_view_idx]
            input_text = nav_action.text if hasattr(nav_action, 'text') else None
            new_action = Utils.pack_action(self.app, action_type=Utils.get_action_type(nav_action), target_element=new_view, input_text=input_text)
            # new_action = copy.deepcopy(nav_action)
            # new_action.view = new_view
            return new_action
        except Exception as e:
            self.logger.warning(f'exception during _get_nav_action: {e}')
            return nav_action

    def parse_log_lines(self):
        log_lines = self.device.logcat.get_recent_lines()
        filtered_lines = []
        app_pid = self.device.get_app_pid(self.app)
        # print(f'current app_pid: {app_pid}')
        for line in log_lines:
            try:
                seps = line.split()
                if int(seps[2]) == app_pid:
                    filtered_lines.append(line)
            except:
                pass
        return filtered_lines

    def get_shortest_nav_steps(self, current_state, target_state):
        normal_nav_steps = self.utg.get_G2_nav_steps(current_state, target_state)
        restart_nav_steps = self.utg.get_G2_nav_steps(self.utg.first_state, target_state)
        normal_nav_steps_len = len(normal_nav_steps) if normal_nav_steps else MAX_NAV_STEPS
        restart_nav_steps_len = len(restart_nav_steps) + 1 if restart_nav_steps else MAX_NAV_STEPS
        if normal_nav_steps_len >= MAX_NAV_STEPS and restart_nav_steps_len >= MAX_NAV_STEPS:
            self.logger.warning(f'get_shortest_nav_steps: cannot find a path to {target_state.structure_str} {target_state.foreground_activity}')
            # # forget the unavailable state
            # target_state_str = target_state.state_str
            # self.memory.known_states.pop(target_state_str)
            # action_strs_to_remove = []
            # for action_str in self.memory.known_transitions:
            #     action_from_state = self.memory.known_transitions[action_str]['from_state']
            #     action_to_state = self.memory.known_transitions[action_str]['to_state']
            #     if action_from_state.state_str == target_state_str or action_to_state.state_str == target_state_str:
            #         action_strs_to_remove.append(action_str)
            # for action_str in action_strs_to_remove:
            #     self.memory.known_transitions.pop(action_str)
            return None
        elif normal_nav_steps_len > restart_nav_steps_len:  # prefer shortest path
        # elif normal_nav_steps_len >= MAX_NAV_STEPS:  # prefer normal navigation
            nav_steps = [(current_state, KillAppEvent(app=self.app))] + restart_nav_steps
        else:
            nav_steps = normal_nav_steps
        return nav_steps
    
    def _dump_memory(self):
        """
        Output current memory to text files
        """
        if not self.device.output_dir:
            return
        if self.action_count % DUMP_MEMORY_NUM_STEPS != 1 and not Manual_mode:
            return
        self.memory.action_history.to_csv(os.path.join(self.device.output_dir, "actions.csv"))
        memory_path = os.path.join(self.device.output_dir, "memory.txt")
        memory_str = self.memory.to_string()
        with open(memory_path, "w") as memory_file:
            memory_file.write(memory_str)
            
    def get_manual_action(self, state):
        
        def debug_action_extract(actions):
            # TODO: add an exit and restart action
            ele_set, action_set, input_set, record_set = False, False, False, False
            element_id, action_choice, input_text_value = None, None, None
            id_states = []

            while not record_set:
                try:
                    response = input(f'\033[0;32mPlease input element ids needed to extract, using " " to separate:\033[0m')
                    if response == '':
                        record_set = True
                        break
                    res = response.strip(" ").split(" ")
                    id_states = []
                    if len(res) != 0:
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
                    record_set = True
                    break
                except KeyboardInterrupt:
                    raise KeyboardInterrupt()
                except:
                    print('warning, wrong format, do not record')
                    continue

            while not ele_set:
                try:
                    response = input(f"Please input element id:")
                    element_id = int(response)
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
                
            if actions[element_id][action_choice] == 'set_text':
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
            return element_id, action_choice, input_text_value, id_states
        
        element_descs, actiontypes, all_elements = self.parse_all_executable_actions(state)
        element_descs_without_bbox = [re.sub(r'\s*bound_box=\d+,\d+,\d+,\d+', '', desc) for desc in element_descs]
        state_desc = "\n".join(element_descs_without_bbox)
        state_desc_with_bbox = "\n".join(element_descs)
        print('='*80, f'\n{state_desc}\n', '='*80)

        id, action_id, input_text, id_states = debug_action_extract(actiontypes)
        selected_action_type, selected_element = actiontypes[id][action_id], all_elements[id]
        
        file_path = os.path.join(self.device.output_dir, 'log.yaml')
        _save2yaml(file_path, state_desc_with_bbox, id, input_text, selected_action_type, state.state_str, state.structure_str, state.tag, state.width, state.height)
        
        # selected
        task_path = os.path.join(self.device.output_dir, f'task_{int(time.time())}.yaml')
        selected_states = None
        if len(id_states) > 0:
            selected_states= '\n'.join([element_descs[i].strip() for i in id_states])
        
        _records2yaml(task_path, state_desc_with_bbox, id, self.app.app_name, selected_states, input_text, selected_action_type, state.state_str, state.structure_str, state.tag, state.width, state.height)
        return Utils.pack_action(self.app, selected_action_type, selected_element, input_text)
        
            
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
        
        

if __name__ == '__main__':
    r = GPT.query('hello!')
    print(r)

