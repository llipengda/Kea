import os
import logging
import random
import copy
import re
import time

import requests

from dotenv import load_dotenv
from .utils import Time, generate_report, save_log, RULE_STATE
from abc import abstractmethod
from .input_event import (
    KEY_RotateDeviceToPortraitEvent,
    KEY_RotateDeviceToLandscapeEvent,
    KeyEvent,
    IntentEvent,
    ReInstallAppEvent,
    RotateDevice,
    RotateDeviceToPortraitEvent,
    RotateDeviceToLandscapeEvent,
    KillAppEvent,
    KillAndRestartAppEvent,
    SetTextEvent, Action, U2Event,
    UIEvent,
)
from .utg import UTG
import json

# from .kea import utils
from .kea import CHECK_RESULT
from typing import TYPE_CHECKING, Dict, Set

if TYPE_CHECKING:
    from .input_manager import InputManager
    from .kea import Kea
    from .app import App
    from .device import Device

# Max number of restarts
MAX_NUM_RESTARTS = 5
# Max number of steps outside the app
MAX_NUM_STEPS_OUTSIDE = 10
MAX_NUM_STEPS_OUTSIDE_KILL = 10
# Max number of replay tries
MAX_REPLY_TRIES = 5
START_TO_GENERATE_EVENT_IN_POLICY = 2
# Max number of query llm
MAX_NUM_QUERY_LLM = 10
START_TO_GENERATE_EVENT_IN_POLICY = 2
# Max number of query llm
MAX_NUM_QUERY_LLM = 10

# Some input event flags
EVENT_FLAG_STARTED = "+started"
EVENT_FLAG_START_APP = "+start_app"
EVENT_FLAG_STOP_APP = "+stop_app"
EVENT_FLAG_EXPLORE = "+explore"
EVENT_FLAG_NAVIGATE = "+navigate"
EVENT_FLAG_TOUCH = "+touch"

# Policy taxanomy
POLICY_GUIDED = "guided"
POLICY_RANDOM = "random"
POLICY_NONE = "none"
POLICY_LLM = "llm"
POLICY_NEW = "new"
POLICY_ENHANCE = "enhance"
POLICY_ONLY_ENHANCE = "only_enhance"

load_dotenv()

GPT_KEY = os.getenv("GPT_KEY")
GPT_URL = os.getenv("GPT_URL")

if not GPT_KEY or not GPT_URL:
    logging.warning("Environment variables GPT_KEY and GPT_URL are not set. ")
else:
    print(GPT_KEY)
    print(GPT_URL)

class InputInterruptedException(Exception):
    pass



class InputPolicy(object):
    """
    This class is responsible for generating events to stimulate more app behaviour
    It should call AppEventManager.send_event method continuously
    """

    def __init__(self, device: "Device", app: "App", allow_to_generate_utg=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.time_recoder = Time()
        self.utg = UTG(device=device, app=app)
        self.device = device
        self.app = app
        self.event_count = 0

        self.last_event = None
        self.from_state = None
        self.to_state = None
        self.allow_to_generate_utg = allow_to_generate_utg
        self.triggered_bug_information = []
        self.time_needed_to_satisfy_precondition = []
        self.statistics_of_rules = {}

        self._num_restarts = 0
        self._num_steps_outside = 0
        self._event_trace = ""

    def start(self, input_manager: "InputManager"):
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        # number of events that have been executed
        self.event_count = 0
        # self.input_manager = input_manager
        while input_manager.enabled and self.event_count < input_manager.event_count:
            try:
                # always try to close the keyboard on the device.
                # if self.device.is_harmonyos is False and hasattr(self.device, "u2"):
                #     self.device.u2.set_fastinput_ime(True)

                self.logger.info("Exploration event count: %d", self.event_count)
                
                if self.to_state is not None:
                    self.from_state = self.to_state
                else:
                    self.from_state = self.device.get_current_state()
                
                # set the from_state to droidbot to let the pdl get the state
                self.device.from_state = self.from_state
                
                if self.event_count == 0:
                    # If the application is running, close the application.
                    event = KillAppEvent(app=self.app)
                elif self.event_count == 1:
                    # start the application
                    event = IntentEvent(self.app.get_start_intent())
                else:
                    event = self.generate_event()

                if event is not None:
                    try:
                        self.device.save_screenshot_for_report(
                            event=event, current_state=self.from_state
                        )
                    except Exception as e:
                        self.logger.error("SaveScreenshotForReport failed: %s", e)
                        self.from_state = self.device.get_current_state()
                        self.device.save_screenshot_for_report(event=event, current_state=self.from_state)
                    input_manager.add_event(event)
                self.to_state = self.device.get_current_state()
                self.last_event = event
                if self.allow_to_generate_utg:
                    self.update_utg()

                bug_report_path = os.path.join(self.device.output_dir, "all_states")
                # TODO this function signature is too long?
                generate_report(
                    bug_report_path,
                    self.device.output_dir,
                    self.triggered_bug_information,
                    self.time_needed_to_satisfy_precondition,
                    self.device.cur_event_count,
                    self.time_recoder.get_time_duration(),
                    self.statistics_of_rules
                )

            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.info("stop sending events: %s" % e)
                self.logger.info("action count: %d" % self.event_count)
                break

            except RuntimeError as e:
                self.logger.info("RuntimeError: %s, stop sending events" % e)
                break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback

                traceback.print_exc()
            self.event_count += 1
        self.tear_down()

    def update_utg(self):
        self.utg.add_transition(self.last_event, self.from_state, self.to_state)

    def move_the_app_to_foreground_if_needed(self, current_state):
        """
        if the app is not running on the foreground of the device, then try to bring it back
        """
        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()

            # It seems the app stucks at some state, has been
            # 1) force stopped (START, STOP)
            #    just start the app again by increasing self.__num_restarts
            # 2) started at least once and cannot be started (START)
            #    pass to let viewclient deal with this case
            # 3) nothing
            #    a normal start. clear self.__num_restarts.

            if self._event_trace.endswith(
                    EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP
            ) or self._event_trace.endswith(EVENT_FLAG_START_APP):
                self._num_restarts += 1
                self.logger.info(
                    "The app had been restarted %d times.", self._num_restarts
                )
            else:
                self._num_restarts = 0

            # pass (START) through
            if not self._event_trace.endswith(EVENT_FLAG_START_APP):
                if self._num_restarts > MAX_NUM_RESTARTS:
                    # If the app had been restarted too many times, enter random mode
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                else:
                    # Start the app
                    self._event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    return IntentEvent(intent=start_app_intent)

        elif current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            self._num_steps_outside += 1

            if self._num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                if self._num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self._event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                return go_back_event
        else:
            # If the app is in foreground
            self._num_steps_outside = 0

    @abstractmethod
    def tear_down(self):
        """ """
        pass

    @abstractmethod
    def generate_event(self):
        """
        generate an event
        @return:
        """
        pass

    @abstractmethod
    def generate_random_event_based_on_current_state(self):
        """
        generate an event
        @return:
        """
        pass


class KeaInputPolicy(InputPolicy):
    """
    state-based input policy
    """

    def __init__(self, device, app, kea: "Kea" = None, allow_to_generate_utg=False):
        super(KeaInputPolicy, self).__init__(device, app, allow_to_generate_utg)
        self.kea = kea
        # self.last_event = None
        # self.from_state = None
        # self.to_state = None

        # retrive all the rules from the provided properties
        for rule in self.kea.all_rules:
            self.statistics_of_rules[str(rule.function.__name__)] = {
                RULE_STATE.PRECONDITION_SATISFIED: 0,
                RULE_STATE.PROPERTY_CHECKED: 0,
                RULE_STATE.POSTCONDITION_VIOLATED: 0,
                RULE_STATE.UI_OBJECT_NOT_FOUND: 0
            }

    def run_initializer(self):
        if self.kea.initializer is None:
            self.logger.warning("No initializer")
            return

        result = self.kea.execute_initializer(self.kea.initializer)
        if (
                result == CHECK_RESULT.PASS
        ):  # why only check `result`, `result` could have different values.
            self.logger.info("-------initialize successfully-----------")
        else:
            self.logger.error("-------initialize failed-----------")

    def check_rule_whose_precondition_are_satisfied(self):
        """
        TODO should split the function
        #! xixian - agree to split the function
        """
        # ! TODO - xixian - should we emphasize the following data structure is a dict?
        rules_ready_to_be_checked = (
            self.kea.get_rules_whose_preconditions_are_satisfied()
        )
        rules_ready_to_be_checked.update(self.kea.get_rules_without_preconditions())
        if len(rules_ready_to_be_checked) == 0:
            self.logger.debug("No rules match the precondition")
            return

        candidate_rules_list = list(rules_ready_to_be_checked.keys())
        # randomly select a rule to check
        rule_to_check = random.choice(candidate_rules_list)

        if rule_to_check is not None:
            self.logger.info(f"-------Check Property : {rule_to_check}------")
            self.statistics_of_rules[str(rule_to_check.function.__name__)][
                RULE_STATE.PROPERTY_CHECKED
            ] += 1
            precondition_page_index = self.device.cur_event_count
            # check rule, record relavant info and output log
            result = self.kea.execute_rule(
                rule=rule_to_check, keaTest=rules_ready_to_be_checked[rule_to_check]
            )
            if result == CHECK_RESULT.ASSERTION_FAILURE:
                self.logger.error(
                    f"-------Postcondition failed. Assertion error, Property:{rule_to_check}------"
                )
                self.logger.debug(
                    "-------time from start : %s-----------"
                    % str(self.time_recoder.get_time_duration())
                )
                self.statistics_of_rules[str(rule_to_check.function.__name__)][
                    RULE_STATE.POSTCONDITION_VIOLATED
                ] += 1
                postcondition_page__index = self.device.cur_event_count
                self.triggered_bug_information.append(
                    (
                        (precondition_page_index, postcondition_page__index),
                        self.time_recoder.get_time_duration(),
                        rule_to_check.function.__name__,
                    )
                )
            elif result == CHECK_RESULT.PASS:
                self.logger.info(
                    f"-------Post condition satisfied. Property:{rule_to_check} pass------"
                )
                self.logger.debug(
                    "-------time from start : %s-----------"
                    % str(self.time_recoder.get_time_duration())
                )

            elif result == CHECK_RESULT.UI_NOT_FOUND:
                self.logger.error(
                    f"-------Execution failed: UiObjectNotFound during exectution. Property:{rule_to_check}-----------"
                )
                self.statistics_of_rules[str(rule_to_check.function.__name__)][
                    RULE_STATE.UI_OBJECT_NOT_FOUND
                ] += 1
            elif result == CHECK_RESULT.PRECON_NOT_SATISFIED:
                self.logger.info("-------Precondition not satisfied-----------")
            else:
                raise AttributeError(f"Invalid property checking result {result}")

    def generate_event(self):
        """
        generate an event
        @return:
        """
        pass

    def update_utg(self):
        self.utg.add_transition(self.last_event, self.from_state, self.to_state)


class RandomPolicy(KeaInputPolicy):
    """
    generate random event based on current app state
    """

    def __init__(
            self,
            device,
            app,
            kea=None,
            restart_app_after_check_property=False,
            number_of_events_that_restart_app=100,
            clear_and_reinstall_app=False,
            allow_to_generate_utg=False,
            disable_rotate=False,
            output_dir=None
    ):
        super(RandomPolicy, self).__init__(device, app, kea, allow_to_generate_utg)
        self.restart_app_after_check_property = restart_app_after_check_property
        self.number_of_events_that_restart_app = number_of_events_that_restart_app
        self.clear_and_reinstall_app = clear_and_reinstall_app
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir=output_dir
        save_log(self.logger, self.output_dir)
        self.disable_rotate=disable_rotate
        self.last_rotate_events = KEY_RotateDeviceToPortraitEvent

    def generate_event(self):
        """
        generate an event
        @return:
        """

        if self.event_count == START_TO_GENERATE_EVENT_IN_POLICY or isinstance(
                self.last_event, ReInstallAppEvent
        ):
            self.run_initializer()
            self.from_state = self.device.get_current_state()
        current_state = self.from_state
        if current_state is None:
            time.sleep(5)
            return KeyEvent(name="BACK")

        if self.event_count % self.number_of_events_that_restart_app == 0:
            if self.clear_and_reinstall_app:
                self.logger.info(
                    "clear and reinstall app after %s events"
                    % self.number_of_events_that_restart_app
                )
                return ReInstallAppEvent(self.app)
            self.logger.info(
                "restart app after %s events" % self.number_of_events_that_restart_app
            )
            return KillAndRestartAppEvent(app=self.app)

        rules_to_check = self.kea.get_rules_whose_preconditions_are_satisfied()
        for rule_to_check in rules_to_check:
            self.statistics_of_rules[str(rule_to_check.function.__name__)][
                RULE_STATE.PRECONDITION_SATISFIED
            ] += 1

        if len(rules_to_check) > 0:
            t = self.time_recoder.get_time_duration()
            self.time_needed_to_satisfy_precondition.append(t)
            self.logger.debug(
                "has rule that matches the precondition and the time duration is "
                + t
            )
            if random.random() < 0.5:
                self.logger.info("Check property")
                self.check_rule_whose_precondition_are_satisfied()
                if self.restart_app_after_check_property:
                    self.logger.debug("restart app after check property")
                    return KillAppEvent(app=self.app)
                return None
            else:
                self.logger.info("Don't check the property due to the randomness")

        event = self.generate_random_event_based_on_current_state()

        return event

    def generate_random_event_based_on_current_state(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        current_state = self.from_state
        self.logger.debug("Current state: %s" % current_state.state_str)
        event = self.move_the_app_to_foreground_if_needed(current_state)
        if event is not None:
            return event

        possible_events = current_state.get_possible_input()
        possible_events.append(KeyEvent(name="BACK"))
        if not self.disable_rotate:
            possible_events.append(RotateDevice())

        self._event_trace += EVENT_FLAG_EXPLORE

        event = random.choice(possible_events)
        if isinstance(event, RotateDevice):
            # select a rotate event with different direction than last time
            if self.last_rotate_events == KEY_RotateDeviceToPortraitEvent:
                self.last_rotate_events = KEY_RotateDeviceToLandscapeEvent
                event = (
                    RotateDeviceToLandscapeEvent()
                )
            else:
                self.last_rotate_events = KEY_RotateDeviceToPortraitEvent
                event = RotateDeviceToPortraitEvent()
        return event


class GuidedPolicy(KeaInputPolicy):
    """
    generate events around the main path
    """

    def __init__(self, device, app, kea=None, allow_to_generate_utg=False,disable_rotate=False,output_dir=None):
        super(GuidedPolicy, self).__init__(device, app, kea, allow_to_generate_utg)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        save_log(self.logger,self.output_dir)
        self.disable_rotate = disable_rotate
        if len(self.kea.all_mainPaths):
            self.logger.info("Found %d mainPaths" % len(self.kea.all_mainPaths))
        else:
            self.logger.error("No mainPath found")

        self.main_path = None
        self.execute_main_path = True

        self.current_index_on_main_path = 0
        self.max_number_of_mutate_steps_on_single_node = 20
        self.current_number_of_mutate_steps_on_single_node = 0
        self.number_of_events_that_try_to_find_event_on_main_path = 0
        self.index_on_main_path_after_mutation = -1
        self.mutate_node_index_on_main_path = 0

        self.last_random_text = None
        self.last_rotate_events = KEY_RotateDeviceToPortraitEvent

    def select_main_path(self):
        if len(self.kea.all_mainPaths) == 0:
            self.logger.error("No mainPath")
            return
        self.main_path = random.choice(self.kea.all_mainPaths)
        # self.path_func, self.main_path =  self.kea.parse_mainPath(self.main_path)
        self.path_func, self.main_path = self.main_path.function, self.main_path.path
        self.logger.info(
            f"Select the {len(self.main_path)} steps mainPath function: {self.path_func}"
        )
        self.main_path_list = copy.deepcopy(self.main_path)
        self.max_number_of_events_that_try_to_find_event_on_main_path = min(
            10, len(self.main_path)
        )
        self.mutate_node_index_on_main_path = len(self.main_path)

    def generate_event(self):
        """ """
        current_state = self.from_state

        # Return relevant events based on whether the application is in the foreground.
        event = self.move_the_app_to_foreground_if_needed(current_state)
        if event is not None:
            return event

        if ((self.event_count == START_TO_GENERATE_EVENT_IN_POLICY)
                or isinstance(self.last_event, ReInstallAppEvent)):
            self.select_main_path()
            self.run_initializer()
            time.sleep(2)
            self.from_state = self.device.get_current_state()
        if self.execute_main_path:
            event_str = self.get_next_event_from_main_path()
            if event_str:
                self.logger.info("*****main path running*****")
                self.kea.execute_event_from_main_path(event_str)
                return None
        if event is None:
            # generate event aroud the state on the main path
            event = self.mutate_the_main_path()

        return event

    def stop_mutation(self):
        self.index_on_main_path_after_mutation = -1
        self.number_of_events_that_try_to_find_event_on_main_path = 0
        self.execute_main_path = True
        self.current_number_of_mutate_steps_on_single_node = 0
        self.current_index_on_main_path = 0
        self.mutate_node_index_on_main_path -= 1
        if self.mutate_node_index_on_main_path == -1:
            self.mutate_node_index_on_main_path = len(self.main_path)
            return ReInstallAppEvent(app=self.app)
        self.logger.info(
            "reach the max number of mutate steps on single node, restart the app"
        )
        return KillAndRestartAppEvent(app=self.app)

    def mutate_the_main_path(self):
        event = None
        self.current_number_of_mutate_steps_on_single_node += 1

        if (
                self.current_number_of_mutate_steps_on_single_node
                >= self.max_number_of_mutate_steps_on_single_node
        ):
            # try to find an event from the main path that can be executed on current state
            if (
                    self.number_of_events_that_try_to_find_event_on_main_path
                    <= self.max_number_of_events_that_try_to_find_event_on_main_path
            ):
                self.number_of_events_that_try_to_find_event_on_main_path += 1
                # if reach the state that satsfies the precondition, check the rule and turn to execute the main path.
                if self.index_on_main_path_after_mutation == len(self.main_path_list):
                    self.logger.info(
                        "reach the end of the main path that could satisfy the precondition"
                    )
                    rules_to_check = self.kea.get_rules_whose_preconditions_are_satisfied()
                    for rule_to_check in rules_to_check:
                        self.statistics_of_rules[str(rule_to_check.function.__name__)][
                            RULE_STATE.PRECONDITION_SATISFIED
                        ] += 1
                    if len(rules_to_check) > 0:
                        t = self.time_recoder.get_time_duration()
                        self.time_needed_to_satisfy_precondition.append(t)
                        self.logger.debug(
                            "has rule that matches the precondition and the time duration is "
                            + t
                        )
                        self.logger.info("Check property")
                        self.check_rule_whose_precondition_are_satisfied()
                    return self.stop_mutation()

                # find if there is any event in the main path that could be executed on currenty state
                event_str = self.get_event_from_main_path()
                try:
                    self.kea.execute_event_from_main_path(event_str)
                    self.logger.info("find the event in the main path")
                    return None
                except Exception:
                    self.logger.info("can't find the event in the main path")
                    return self.stop_mutation()

            return self.stop_mutation()

        self.index_on_main_path_after_mutation = -1

        if len(self.kea.get_rules_whose_preconditions_are_satisfied()) > 0:
            # if the property has been checked, don't return any event
            t = self.time_recoder.get_time_duration()
            self.time_needed_to_satisfy_precondition.append(t)
            self.logger.debug(
                "has rule that matches the precondition and the time duration is "
                + t
            )
            if random.random() < 0.5:
                self.logger.info("Check property")
                self.check_rule_whose_precondition_are_satisfied()
                return None
            else:
                self.logger.info("Don't check the property due to the randomness")

        event = self.generate_random_event_based_on_current_state()
        return event

    def get_next_event_from_main_path(self):
        """
        get a next event when execute on the main path
        """
        if self.current_index_on_main_path == self.mutate_node_index_on_main_path:
            self.logger.info(
                "reach the mutate index, start mutate on the node %d"
                % self.mutate_node_index_on_main_path
            )
            self.execute_main_path = False
            return None

        self.logger.info(
            "execute node index on main path: %d" % self.current_index_on_main_path
        )
        u2_event_str = self.main_path_list[self.current_index_on_main_path]
        if u2_event_str is None:
            self.logger.warning(
                "event is None on main path node %d" % self.current_index_on_main_path
            )
            self.current_index_on_main_path += 1
            return self.get_next_event_from_main_path()
        self.current_index_on_main_path += 1
        return u2_event_str

    def get_ui_element_dict(self, ui_element_str: str) -> Dict[str, str]:
        """
        get ui elements of the event
        """
        start_index = ui_element_str.find("(") + 1
        end_index = ui_element_str.find(")", start_index)

        if start_index != -1 and end_index != -1:
            ui_element_str = ui_element_str[start_index:end_index]
        ui_elements = ui_element_str.split(",")

        ui_elements_dict = {}
        for ui_element in ui_elements:
            attribute_name, attribute_value = ui_element.split("=")
            attribute_name = attribute_name.strip()
            attribute_value = attribute_value.strip()
            attribute_value = attribute_value.strip('"')
            ui_elements_dict[attribute_name] = attribute_value
        return ui_elements_dict

    def get_event_from_main_path(self):
        """
        get an event can lead current state to go back to the main path
        """
        if self.index_on_main_path_after_mutation == -1:
            for i in reversed(range(len(self.main_path_list))):
                event_str = self.main_path_list[i]
                ui_elements_dict = self.get_ui_element_dict(event_str)
                current_state = self.from_state
                view = current_state.get_view_by_attribute(ui_elements_dict)
                if view is None:
                    continue
                self.index_on_main_path_after_mutation = i + 1
                return event_str
        else:
            event_str = self.main_path_list[self.index_on_main_path_after_mutation]
            ui_elements_dict = self.get_ui_element_dict(event_str)
            current_state = self.from_state
            view = current_state.get_view_by_attribute(ui_elements_dict)
            if view is None:
                return None
            self.index_on_main_path_after_mutation += 1
            return event_str
        return None

    def generate_random_event_based_on_current_state(self):
        """
        generate an event based on current UTG to explore the app
        @return: InputEvent
        """
        current_state = self.from_state
        self.logger.info("Current state: %s" % current_state.state_str)
        event = self.move_the_app_to_foreground_if_needed(current_state)
        if event is not None:
            return event

        # Get all possible input events
        possible_events = current_state.get_possible_input()

        # if self.random_input:
        #     random.shuffle(possible_events)
        possible_events.append(KeyEvent(name="BACK"))
        if not self.disable_rotate:
            possible_events.append(RotateDevice())

        self._event_trace += EVENT_FLAG_EXPLORE

        event = random.choice(possible_events)
        if isinstance(event, RotateDevice):
            if self.last_rotate_events == KEY_RotateDeviceToPortraitEvent:
                self.last_rotate_events = KEY_RotateDeviceToLandscapeEvent
                event = RotateDeviceToLandscapeEvent()
            else:
                self.last_rotate_events = KEY_RotateDeviceToPortraitEvent
                event = RotateDeviceToPortraitEvent()

        return event


class LLMPolicy(RandomPolicy):
    """
    use LLM to generate input when detected ui tarpit
    """

    def __init__(
            self,
            device,
            app,
            kea=None,
            restart_app_after_check_property=False,
            number_of_events_that_restart_app=100,
            clear_and_restart_app_data_after_100_events=False,
            allow_to_generate_utg=False,
            output_dir=None
    ):
        super(LLMPolicy, self).__init__(device, app, kea, output_dir=output_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        save_log(self.logger,self.output_dir)
        self.__action_history = []
        self.__all_action_history = set()
        self.__activity_history = set()
        self.from_state = None
        self.task = "You are an expert in App GUI testing. Please guide the testing tool to enhance the coverage of functional scenarios in testing the App based on your extensive App testing experience. "

    def start(
            self, input_manager: "InputManager"
    ):  # TODO do not need to write start here?
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        self.event_count = 0
        self.input_manager = input_manager
        while input_manager.enabled and self.event_count < input_manager.event_count:
            try:
                if self.device.is_harmonyos == False and hasattr(self.device, "u2"):
                    self.device.u2.set_fastinput_ime(True)

                self.logger.info("Exploration action count: %d" % self.event_count)

                # if self.to_state is not None:
                #     self.from_state = self.to_state
                # else:
                #     self.from_state = self.device.get_current_state()

                if self.event_count == 0:
                    # If the application is running, close the application.
                    event = KillAppEvent(app=self.app)
                elif self.event_count == 1:
                    event = IntentEvent(self.app.get_start_intent())
                else:
                    if input_manager.sim_calculator.detected_ui_tarpit(input_manager):
                        # If detected a ui tarpit
                        if input_manager.sim_calculator.sim_count > MAX_NUM_QUERY_LLM:
                            # If query LLM too much
                            self.logger.info(f"query too much. go back!")
                            event = KeyEvent(name="BACK")
                            self.clear_action_history()
                            input_manager.sim_calculator.sim_count = 0
                        else:
                            # stop random policy, start query LLM
                            event = self.generate_llm_event()
                    else:
                        event = self.generate_event()
                        
                self.from_state = self.device.get_current_state()

                if event is not None:
                    self.device.save_screenshot_for_report(
                        event=event, current_state=self.from_state
                    )
                    input_manager.add_event(event)
                self.to_state = self.device.get_current_state()
                self.last_event = event
                if self.allow_to_generate_utg:
                    self.update_utg()

                bug_report_path = os.path.join(self.device.output_dir, "all_states") #type: ignore
                generate_report(
                    bug_report_path,
                    self.device.output_dir,
                    self.triggered_bug_information,
                    self.time_needed_to_satisfy_precondition,
                    self.device.cur_event_count,
                    self.time_recoder.get_time_duration(),
                )
            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.info("stop sending events: %s" % e)
                self.logger.info("action count: %d" % self.event_count)
                break

            except RuntimeError as e:
                self.logger.info("RuntimeError: %s, stop sending events" % e)
                break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback

                traceback.print_exc()
            self.event_count += 1
        self.tear_down()

    def generate_llm_event(self):
        """
        generate an LLM event
        @return:
        """

        if self.event_count == START_TO_GENERATE_EVENT_IN_POLICY or isinstance(
                self.last_event, ReInstallAppEvent
        ):
            self.run_initializer()
            self.from_state = self.device.get_current_state()
        current_state = self.from_state
        if current_state is None:
            import time

            time.sleep(5)
            return KeyEvent(name="BACK")

        if (
                self.event_count % self.number_of_events_that_restart_app == 0
                and self.clear_and_reinstall_app
        ):
            self.logger.info(
                "clear and restart app after %s events"
                % self.number_of_events_that_restart_app
            )
            return ReInstallAppEvent(self.app)
        rules_to_check = self.kea.get_rules_whose_preconditions_are_satisfied()
        for rule_to_check in rules_to_check:
            self.statistics_of_rules[str(rule_to_check.function.__name__)][
                RULE_STATE.PRECONDITION_SATISFIED
            ] += 1

        if len(rules_to_check) > 0:
            t = self.time_recoder.get_time_duration()
            self.time_needed_to_satisfy_precondition.append(t)
            self.logger.debug(
                "has rule that matches the precondition and the time duration is "
                + self.time_recoder.get_time_duration()
            )
            if random.random() < 0.5:
                self.logger.info("Check property")
                self.check_rule_whose_precondition_are_satisfied()
                if self.restart_app_after_check_property:
                    self.logger.debug("restart app after check property")
                    return KillAppEvent(app=self.app)
                return None
            else:
                self.logger.info(
                    "Found exectuable property in current state. No property will be checked now according to the random checking policy."
                )
        event = None

        if event is None:
            event = self.generate_llm_event_based_on_utg()

        if isinstance(event, RotateDevice):
            if self.last_rotate_events == KEY_RotateDeviceToPortraitEvent:
                self.last_rotate_events = KEY_RotateDeviceToLandscapeEvent
                event = RotateDeviceToLandscapeEvent()
            else:
                self.last_rotate_events = KEY_RotateDeviceToPortraitEvent
                event = RotateDeviceToPortraitEvent()

        return event

    def generate_llm_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        current_state = self.from_state
        self.logger.info("Current state: %s" % current_state.state_str)

        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()

            # It seems the app stucks at some state, has been
            # 1) force stopped (START, STOP)
            #    just start the app again by increasing self.__num_restarts
            # 2) started at least once and cannot be started (START)
            #    pass to let viewclient deal with this case
            # 3) nothing
            #    a normal start. clear self.__num_restarts.

            if self._event_trace.endswith(
                    EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP
            ) or self._event_trace.endswith(EVENT_FLAG_START_APP):
                self._num_restarts += 1
                self.logger.info(
                    "The app had been restarted %d times.", self._num_restarts
                )
            else:
                self._num_restarts = 0

            # pass (START) through
            if not self._event_trace.endswith(EVENT_FLAG_START_APP):
                if self._num_restarts > MAX_NUM_RESTARTS:
                    # If the app had been restarted too many times, enter random mode
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                    self.__random_explore = True
                else:
                    # Start the app
                    self._event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    self.__action_history = [f"- start the app {self.app.app_name}"]
                    return IntentEvent(intent=start_app_intent)

        elif current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            self.__num_steps_outside += 1

            if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self._event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                self.__action_history.append("- go back")
                return go_back_event
        else:
            # If the app is in foreground
            self.__num_steps_outside = 0

        action, candidate_actions = self._get_action_with_LLM(
            current_state,
            self.__action_history,
            self.__activity_history,
        )
        if action is not None:
            self.__action_history.append(current_state.get_action_desc(action))
            self.__all_action_history.add(current_state.get_action_desc(action))
            return action

        if self.__random_explore:
            self.logger.info("Trying random event...")
            action = random.choice(candidate_actions)
            self.__action_history.append(current_state.get_action_desc(action))
            self.__all_action_history.add(current_state.get_action_desc(action))
            return action

        # If couldn't find a exploration target, stop the app
        stop_app_intent = self.app.get_stop_intent()
        self.logger.info("Cannot find an exploration target. Trying to restart app...")
        self.__action_history.append("- stop the app")
        self.__all_action_history.add("- stop the app")
        self._event_trace += EVENT_FLAG_STOP_APP
        return IntentEvent(intent=stop_app_intent)

    def _query_llm(self, prompt, model_name="gpt-3.5-turbo"):
        # TODO: replace with your own LLM
        from openai import OpenAI

        gpt_url = GPT_URL
        gpt_key = GPT_KEY
        client = OpenAI(base_url=gpt_url, api_key=gpt_key)

        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            messages=messages, model=model_name, timeout=30
        )
        res = completion.choices[0].message.content
        return res

    def _get_action_with_LLM(self, current_state, action_history, activity_history):
        activity = current_state.foreground_activity
        task_prompt = (
                self.task
                + f"Currently, the App is stuck on the {activity} page, unable to explore more features. You task is to select an action based on the current GUI Infomation to perform next and help the app escape the UI tarpit."
        )
        visisted_page_prompt = (
                f"I have already visited the following activities: \n"
                + "\n".join(activity_history)
        )
        # all_history_prompt = f'I have already completed the following actions to explore the app: \n' + '\n'.join(all_action_history)
        history_prompt = (
                f"I have already completed the following steps to leave {activity} page but failed: \n "
                + ";\n ".join(action_history)
        )
        state_prompt, candidate_actions = current_state.get_described_actions()
        question = "Which action should I choose next? Just return the action id and nothing else.\nIf no more action is needed, return -1."
        prompt = f"{task_prompt}\n{state_prompt}\n{visisted_page_prompt}\n{history_prompt}\n{question}"
        print(prompt)
        response = self._query_llm(prompt)
        print(f"response: {response}")

        match = re.search(r"\d+", response)
        if not match:
            return None, candidate_actions
        idx = int(match.group(0))
        selected_action = candidate_actions[idx]
        if isinstance(selected_action, SetTextEvent):
            view_text = current_state.get_view_desc(selected_action.view)
            question = f"What text should I enter to the {view_text}? Just return the text and nothing else."
            prompt = f"{task_prompt}\n{state_prompt}\n{question}"
            print(prompt)
            response = self._query_llm(prompt)
            print(f"response: {response}")
            selected_action.text = response.replace('"', "")
            if len(selected_action.text) > 30:  # heuristically disable long text input
                selected_action.text = ""
        return selected_action, candidate_actions

    def get_last_state(self):
        return self.from_state

    def clear_action_history(self):
        self.__action_history = []


import xml.etree.ElementTree as ET
from openai import OpenAI
from .utils import get_xml

class NewPolicy(RandomPolicy):
    def __init__(
            self,
            device: "Device",
            app: "App",
            kea: "Kea" = None,
            restart_app_after_check_property=False,
            number_of_events_that_restart_app=100,
            clear_and_restart_app_data_after_100_events=False,
            allow_to_generate_utg=False,
            disable_rotate=False,
            output_dir: str = None,
    ):
        super(NewPolicy, self).__init__(device, app, kea, output_dir=output_dir,
                                        restart_app_after_check_property=restart_app_after_check_property,
                                        number_of_events_that_restart_app=number_of_events_that_restart_app,
                                        clear_and_reinstall_app=clear_and_restart_app_data_after_100_events,
                                        allow_to_generate_utg=allow_to_generate_utg,
                                        disable_rotate=disable_rotate)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        save_log(self.logger, self.output_dir)
        self.input_manager: "InputManager | None" = None
        self._messages = []
        gpt_url = GPT_URL
        gpt_key = GPT_KEY
        self.client = OpenAI(base_url=gpt_url, api_key=gpt_key)
        self._xml1 = ""
        self._xml2 = ""
        self._in_llm = False
        self._generated_tasks = set()
        self._llm_cnt = 0
        self._out_cnt = 0

    def start(self, input_manager: "InputManager"):
        self.event_count = 0
        self.input_manager = input_manager
        while input_manager.enabled and self.event_count < input_manager.event_count:
            try:
                if self.device.is_harmonyos == False and hasattr(self.device, "u2"):
                    self.device.u2.set_fastinput_ime(True)

                self.logger.info("Exploration action count: %d" % self.event_count)

                if self.to_state is not None:
                    self.from_state = self.to_state
                else:
                    self.from_state = self.device.get_current_state()
                
                self.device.from_state = self.from_state

                self._xml2 = self._xml1
                self._xml1 = get_xml(self.device.u2)

                if self.event_count == 0:
                    # If the application is running, close the application.
                    event = KillAppEvent(app=self.app)
                elif self.event_count == 1:
                    event = IntentEvent(self.app.get_start_intent())
                elif self.event_count % 600 == 0:
                    if self.clear_and_reinstall_app:
                        self.logger.info(
                            "clear and reinstall app after %s events"
                            % self.number_of_events_that_restart_app
                        )
                        event = ReInstallAppEvent(self.app)
                    else:
                        self.logger.info(
                            "restart app after %s events" % self.number_of_events_that_restart_app
                        )
                        event = KillAndRestartAppEvent(app=self.app)
                    self._generated_tasks.clear()
                else:
                    # event = self.move_the_app_to_foreground_if_needed(self.device.get_current_state())
                    self.move_if_need()
                    if self._in_llm:
                        if self._llm_cnt > 8:
                            self._llm_cnt = 0
                            event = KeyEvent(name="BACK")
                            self._in_llm = False
                        else:
                            self._llm_cnt += 1
                            event = self.generate_llm_event()
                    elif input_manager.sim_calculator.detect(self._xml1, self._xml2):
                        # If detected a ui tarpit
                        # if input_manager.sim_calculator.sim_count > MAX_NUM_QUERY_LLM:
                        #     # If query LLM too much
                        #     self.logger.info(f"query too much. go back!")
                        #     event = KeyEvent(name="BACK")
                        #     input_manager.sim_calculator.sim_count = 0
                        # else:
                        #     # stop random policy, start query LLM
                        event = self.generate_llm_event()
                    else:
                        event = self.generate_event()

                self.process_event(event, input_manager)

            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.info("stop sending events: %s" % e)
                self.logger.info("action count: %d" % self.event_count)
                break

            except RuntimeError as e:
                self.logger.info("RuntimeError: %s, stop sending events" % e)
                break

            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback
                traceback.print_exc()

        self.tear_down()
        
    def generate_event(self):
        """
        generate an event
        @return:
        """

        if self.event_count == START_TO_GENERATE_EVENT_IN_POLICY or isinstance(
                self.last_event, ReInstallAppEvent
        ):
            self.run_initializer()
            self.from_state = self.device.get_current_state()
        current_state = self.from_state
        if current_state is None:
            time.sleep(5)
            return KeyEvent(name="BACK")

        rules_to_check = self.kea.get_rules_whose_preconditions_are_satisfied()
        for rule_to_check in rules_to_check:
            self.statistics_of_rules[str(rule_to_check.function.__name__)][
                RULE_STATE.PRECONDITION_SATISFIED
            ] += 1

        if len(rules_to_check) > 0:
            t = self.time_recoder.get_time_duration()
            self.time_needed_to_satisfy_precondition.append(t)
            self.logger.debug(
                "has rule that matches the precondition and the time duration is "
                + t
            )
            if random.random() < 0.5:
                self.logger.info("Check property")
                self.check_rule_whose_precondition_are_satisfied()
                if self.restart_app_after_check_property:
                    self.logger.debug("restart app after check property")
                    return KillAppEvent(app=self.app)
                return None
            else:
                self.logger.info("Don't check the property due to the randomness")

        event = self.generate_random_event_based_on_current_state()

        return event
    
    def generate_random_event_based_on_current_state(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        current_state = self.from_state
        self.logger.debug("Current state: %s" % current_state.state_str)
        possible_events = current_state.get_possible_input()
        possible_events.append(KeyEvent(name="BACK"))
        if not self.disable_rotate:
            possible_events.append(RotateDevice())

        self._event_trace += EVENT_FLAG_EXPLORE

        event = random.choice(possible_events)
        if isinstance(event, RotateDevice):
            # select a rotate event with different direction than last time
            if self.last_rotate_events == KEY_RotateDeviceToPortraitEvent:
                self.last_rotate_events = KEY_RotateDeviceToLandscapeEvent
                event = (
                    RotateDeviceToLandscapeEvent()
                )
            else:
                self.last_rotate_events = KEY_RotateDeviceToPortraitEvent
                event = RotateDeviceToPortraitEvent()
        return event

    def process_event(self, event, input_manager):
        if event is not None:
            try:
                self.device.save_screenshot_for_report(
                    event=event, current_state=self.from_state
                )
            except:
                self.from_state = self.device.get_current_state()
                self.device.save_screenshot_for_report(event=event, current_state=self.from_state)
            finally:
                input_manager.add_event(event)
                self.event_count += 1
        
        self.last_event = event
        self.to_state = self.device.get_current_state()
        if self.allow_to_generate_utg:
            self.update_utg()
        bug_report_path = os.path.join(self.device.output_dir, "all_states")
        generate_report(
            bug_report_path,
            self.device.output_dir,
            self.triggered_bug_information,
            self.time_needed_to_satisfy_precondition,
            self.device.cur_event_count,
            self.time_recoder.get_time_duration(),
        )

    def generate_llm_event(self):
        if self._in_llm:
            self.action_prompt3()
            self.llm()
            self.check_prompt()
            act = Action(**json.loads(str(self.llm().content)))
            self._in_llm = act.hasNext
            return U2Event(act)
        self._messages = []
        self.meaning_prompt()
        self.llm()
        self.action_prompt1()
        res = self.llm().content
        self._generated_tasks.add(res.split('\n')[0])
        self.action_prompt2()
        self.llm()
        self.check_prompt()
        act = Action(**json.loads(self.llm().content))
        self._in_llm = act.hasNext
        return U2Event(act)

    # def generate_llm_event(self):
    #     self._messages = []
    #     self.meaning_prompt()
    #     self.llm()
    #     self.action_prompt()
    #     self.llm()
    #     self.check_prompt()
    #     response = self.llm()
    #     events = [U2Event(Action(**i)) for i in json.loads(str(response.content))]
    #
    #     for e in events:
    #         try:
    #             self.process_event(e, self.input_manager)
    #         except Exception as e:
    #             self.logger.warning(e)
    #             self.error_prompt()
    #             response = self.llm()
    #             new_events = [U2Event(Action(**i)) for i in json.loads(str(response.content))]
    #             for new_event in new_events:
    #                 try:
    #                     self.process_event(new_event, self.input_manager)
    #                 except Exception as e:
    #                     self.logger.warning(e)
    #                     self.process_event(KeyEvent(name="BACK"), self.input_manager)
    #                     break
    #             break
    #
    #     self.recheck_prompt()
    #     response = self.llm()
    #
    #     ok, res = self.parse_recheck(str(response.content))
    #
    #     if not ok:
    #         events = [U2Event(Action(**i)) for i in json.loads(str(res))]
    #         for e in events:
    #             try:
    #                 self.process_event(e, self.input_manager)
    #             except Exception as e:
    #                 self.logger.warning(e)
    #                 self.error_prompt()
    #                 response = self.llm()
    #                 new_events = [U2Event(Action(**i)) for i in json.loads(str(response.content))]
    #                 for new_event in new_events:
    #                     try:
    #                         self.process_event(new_event, self.input_manager)
    #                     except Exception as e:
    #                         self.logger.warning(e)
    #                         self.process_event(KeyEvent(name="BACK"), self.input_manager)
    #                         break
    #                 break

    # def get_xml(self):
    #     d = self.device.u2
    #     root = ET.fromstring(d.dump_hierarchy())
    #
    #     flag = False
    #     for child in root:
    #         for child_child in child:
    #             if child_child.attrib['resource-id'] == 'com.android.systemui:id/status_bar_container':
    #                 root.remove(child)
    #                 flag = True
    #                 break
    #         if flag:
    #             break
    #
    #     def clean_element(element):
    #         for attr in list(element.attrib):
    #             if element.attrib[attr] == "" or element.attrib[attr] == "false":
    #                 del element.attrib[attr]
    #         for child in element:
    #             clean_element(child)
    #
    #     clean_element(root)
    #
    #     res = ET.tostring(root, encoding='unicode')
    #     res = res.replace("content-desc", "description")
    #     return res

    def meaning_prompt(self):
        prompt = f"""This is an XML representation of an Android application page:
    {get_xml(self.device.u2)}
    Please describe the purpose of this page in the most concise language possible.
    """
        self._messages.append({"role": "user", "content": prompt})

    # %%
    # def action_prompt(self):
    #     prompt = f"""If you were the user, what would you do on this page?
    # Please provide an action or a sequence of actions in JSON format, for example:
    # [{{
    #     "action": "click",
    #     "selectors": {{"resourceId": "com.example:id/button1"}}
    # }},
    # {{
    #     "action": "input_text",
    #     "selectors": {{"resourceId": "com.example:id/input", "text": "password"}}
    #     "inputText": "123456"
    # }}]
    # Where:
    # - action can only be one of: click, long_click, input_text, press, swipe, scroll
    # - selector can only be one of: text, className, description, resourceId (must be in camelCase); choose the selector that uniquely identifies the element
    # - the selector's value and must be found in the provided XML
    # - inputText is the input text, applicable only when action is input_text
    # - pressKey can be "enter" and applicable only if action is "press"
    #
    # Please combine multiple selectors to ensure uniquely locating an element.
    #
    # Before outputting, check whether the value exists in the XML. If it does not exist, modify the action accordingly.
    #
    # Return only the JSON-formatted action sequence, without explanations or code blocks.
    # """
    #     self._messages.append({"role": "user", "content": prompt})

    def action_prompt1(self):
        prompt = f"""If you were the user, what would you do on this page? You can only describe one action. 
    Please try to generate tasks that have not been generated before. Below are the tasks that have already been generated:
    {list(self._generated_tasks)}
    Please list the steps required to complete this action. (This action will be named 'The Task')
    Note: You can directly input text to an input box, without clicking it first.
    Note: If there is a drawer, navigate by it might be a good choice.
    DONT GENERATE A TASK THAT MAY LEAVE THE APP.
    """

        self._messages.append({"role": "user", "content": prompt})

    def action_prompt2(self):
        prompt = """Please describe the **first step** of the operation you just performed in JSON format, as shown below:
    {
        "action": "input_text",
        "selectors": {"resourceId": "com.example:id/input", "text": "password"},
        "inputText": "123456",
        "hasNext": true
    }
    Notes:
    - The "action" must be one of: click, long_click, input_text, press_enter
    - "selectors" can only include: **text**, **className**, **description**, **resourceId**, and must be in camelCase. You can not use other selectors.
    - The value is the value of the selector, which must be found in the previous XML
    - "inputText" is the text to input, only present when the action is input_text
    - "hasNext" is a boolean indicating whether there is a next step. Set it to false if there is no next step
    Try to combine multiple selectors to uniquely identify the element.
    Please return the operation in JSON format only. Do not explain or use code blocks.
    """
        self._messages.append({"role": "user", "content": prompt})

    def action_prompt3(self):
        prompt = f"""Now, the current state of the page is as follows: {get_xml(self.device.u2)}
    Please describe the **next step** of the operation you just performed in JSON format, using the same format as above.
    """
        self._messages.append({"role": "user", "content": prompt})

    # def check_prompt(self):
    #     prompt = f"""Now, the current state of the page is: {get_xml(self.device.u2)}
    # Please check: have **all the steps** of "The Task" been completed?
    # """
    #     self._messages.append({"role": "user", "content": prompt})

    def check_prompt(self):
        prompt = """Please check whether the operation or sequence of operations you just generated meets the requirements:
    - The selectors must be found in the XML.
    - The selectors must uniquely identify the element.
    If there are no issues, output it as is; otherwise, modify it accordingly.
    Output in JSON format only, with explanations as a new field "explanation".
    Don't use code blocks.
    """
        self._messages.append({"role": "user", "content": prompt})

    # %%
    # def recheck_prompt(self):
    #     prompt = f"""Please determine whether the current page is very very similar to the previously displayed XML page.
    # {self.get_xml()}
    # If it is very very similar, find a way to escape from the page, output "YES", and then output the operation sequence according to the previous rules.
    # If it is not similar, output "NO".
    # """
    #     self._messages.append({"role": "user", "content": prompt})
    #
    # # %%
    # @staticmethod
    # def parse_recheck(message: str):
    #     if (message.startswith("YES")):
    #         _message = message[3:]
    #         return False, _message
    #     return True, None

    # %%
    def error_prompt(self):
        prompt = f"""
    The event sequence you generated encountered an error because the corresponding element could not be found. This is the current XML representation of the application. 
    {get_xml(self.device.u2)}
    Correct this error. Output the operation sequence according to the previous rules.
    """
        self._messages.append({"role": "user", "content": prompt})

    def llm(self):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self._messages,
        )
        self.logger.info('>' * 60)
        self.logger.info('\n' + self._messages[-1]['content'])
        self.logger.info('<' * 60)
        self.logger.info('\n' + response.choices[0].message.content)
        self._messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message

    def get_last_state(self):
        return self.from_state

    def clear_action_history(self):
        pass

    def check_is_app(self):
        package_name = self.app.get_package_name()
        xml = get_xml(self.device.u2)
        if package_name in xml:
            return True
        else:
            return False

    def get_top(self):
        res = self.device.adb.shell("dumpsys activity top | grep ACTIVITY")
        package_names = re.findall(r'ACTIVITY\s+([^\s/]+)/', res)
        self.logger.info("current package name: %s" % package_names[-1])
        return package_names[-1]

    def move_if_need(self):
        app_name = self.app.get_package_name()
        if self.from_state is not None and app_name in self.from_state.foreground_activity:
            return
        top = self.get_top()
        self.logger.info("top activity: %s; out cnt: %s; app: %s" % (top, self._out_cnt, self.app.get_package_name()))
        if (top) != self.app.get_package_name():
            self._in_llm = False
            self._llm_cnt = 0
            if (top == 'com.google.android.apps.nexuslauncher'):
                self.device.adb.shell(self.app.get_start_intent().get_cmd())
                return
            self._out_cnt += 1
            if self._out_cnt > 2:
                self._out_cnt = 0
                self.logger.info("move the app to foreground")
                self.device.u2.press("BACK")
                time.sleep(0.2)
                top = self.get_top()
                if top != self.app.get_package_name():
                    if (top != 'com.google.android.apps.nexuslauncher'):
                        self.device.adb.shell("am force-stop %s" % top)
                    time.sleep(0.2)
                    top = self.get_top()
                    self.logger.info("now top activity: %s" % top)
                    if top != self.app.get_package_name():
                        self.device.adb.shell("am force-stop %s" % self.app.get_package_name())
                        self.device.adb.shell(self.app.get_start_intent().get_cmd())
        else:
            self._out_cnt = 0

class TestContext:
    def __init__(self):
        self.stack = []
        self.variables = {}

    def push_goal(self, goal):
        self.stack.append({
            "target": goal,
            "status": "pending",
            "dependencies": []
        })

    def mark_completed(self, goal):
        for item in self.stack:
            if item["target"] == goal:
                item["status"] = "completed"

    def get_current_task(self):
        return next((item for item in reversed(self.stack)
                     if item["status"] == "pending"), None)

def encode_image(path):
    with open(path, "rb") as image_file:
        import base64
        return base64.b64encode(image_file.read()).decode("utf-8")

def query_llm(role, content, image_path = None):
    url = None
    api_key = None

    if image_path:
        base64_image = encode_image(image_path)
        payload = {
            "messages": [
                {"role": "system", "content": role},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "stream": False,
            "model": "ecnu-vl"
        }
    else:
        payload = {
            "messages": [
                {"role": "system", "content": role},
                {
                    "role": "user",
                    "content": content
                }],
            "stream": False,
            "model": "ecnu-plus"
        }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, json=payload, headers=headers)
    res = response.json()
    if 'choices' not in res or 'message' not in res['choices'][0] or 'content' not in res['choices'][0]['message']:
        print("QUERY LLM ERROR")
        print(res)
        time.sleep(3)
        return None
    res = response.json()["choices"][0]["message"]["content"]

    return res

def clean_text(text):
    text = text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r'[\r\n\t]', '', text)
    return text

def extract_and_validate_json(text):
    text = clean_text(text)

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        return None

    json_str = json_match.group(0)

    try:
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError:
        return None

def get_event_str(event, state):
    if isinstance(event, RotateDevice) or isinstance(event, KeyEvent) or not hasattr(event, "view"):
        event_str = event.get_event_name()
    else:
        view_name = UIEvent.view_short_str(state, event.view)
        x, y = state.get_view_center(event.view)
        event_str = f"{event.get_event_name()}({view_name}, x={x}, y={y})"
    return event_str

MAX_SAME_EVENT_COUNT = 5

class EnhancedNewPolicy(NewPolicy):
    def __init__(
            self,
            device: "Device",
            app: "App",
            kea: "Kea" = None,
            restart_app_after_check_property=False,
            number_of_events_that_restart_app=100,
            clear_and_restart_app_data_after_100_events=False,
            allow_to_generate_utg=False,
            output_dir: str = None,
            decay_factor=0.8,
            disable_rotate=False,
    ):
        super(EnhancedNewPolicy, self).__init__(device, app, kea, output_dir=output_dir,
                                        restart_app_after_check_property=restart_app_after_check_property,
                                        number_of_events_that_restart_app=number_of_events_that_restart_app,
                                        clear_and_restart_app_data_after_100_events=clear_and_restart_app_data_after_100_events,
                                        allow_to_generate_utg=allow_to_generate_utg, disable_rotate=disable_rotate)
        self.decay_factor = decay_factor
        self.init_utg = None
        self.history = []
        self.last_state = None
        self.input_table = {}
        self.event_table = {}
        # FIXME True 增强的随机策略 False 全程大模型引导
        self.random_test = True
        if not self.random_test:
            self.task_stack = TestContext()

    def start(self, input_manager: "InputManager"):
        if not self.random_test:
            precond_code = ""
            for k, v in self.kea.all_rules_DB.items():
                precond = k.preconditions[0]
                if hasattr(precond, "__source__"):
                    precond_code = precond.__source__

            # TODO Maybe too concise to be located for LLM
            init_task_prompt = f"""You are an expert in mobile app test automation. Convert the given UI element selector into a natural language test precondition statement by following these rules:

    1. Extract the keyword after the last "/" in resourceId
    2. Paraphrase the keyword into a natural language description while preserving its core meaning
    3. Describe the expected position or state of the element
    4. Output the result in natural language, including technical context or clues
    5. If input is invalid/empty/unparseable, return:
    "Proceed with free exploration and maximize code coverage"
    6. Format output as:
    "The [element_type] for [function] should be [state] [position_context]"

    Input:
    {precond_code}"""

            init_task = query_llm("You are an expert in mobile app test automation.", init_task_prompt)
            self.task_stack.push_goal(init_task)
            self.logger.info(f"init task: {init_task}")
        
        self.event_count = 0
        self.input_manager = input_manager
        while input_manager.enabled and self.event_count < input_manager.event_count:
            try:
                if self.device.is_harmonyos == False and hasattr(self.device, "u2"):
                    self.device.u2.set_fastinput_ime(True)

                self.logger.info("Exploration action count: %d" % self.event_count)

                if self.to_state is not None:
                    self.from_state = self.to_state
                else:
                    self.from_state = self.device.get_current_state()
                    
                self.device.from_state = self.from_state

                self._xml2 = self._xml1
                self._xml1 = get_xml(self.device.u2)

                if self.event_count == 0:
                    event = KillAppEvent(app=self.app)
                elif self.event_count == 1:
                    event = IntentEvent(self.app.get_start_intent())
                elif self.event_count % 600 == 0:
                    if self.clear_and_reinstall_app:
                        self.logger.info(
                            "clear and reinstall app after %s events"
                            % self.number_of_events_that_restart_app
                        )
                        event = ReInstallAppEvent(self.app)
                    else:
                        self.logger.info(
                            "restart app after %s events" % self.number_of_events_that_restart_app
                        )
                        event = KillAndRestartAppEvent(app=self.app)
                    self._generated_tasks.clear()
                else:
                    self.move_if_need()
                    if self._in_llm:
                        if self._llm_cnt > 8:
                            self._llm_cnt = 0
                            event = KeyEvent(name="BACK")
                            self._in_llm = False
                        else:
                            self._llm_cnt += 1
                            event = self.generate_llm_event()
                    elif input_manager.sim_calculator.detect(self._xml1, self._xml2):
                        event = self.generate_llm_event()
                    else:
                        event = self.generate_event()

                self.process_event(event, input_manager)

            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.info("stop sending events: %s" % e)
                self.logger.info("action count: %d" % self.event_count)
                break

            except RuntimeError as e:
                self.logger.info("RuntimeError: %s, stop sending events" % e)
                break

            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback
                traceback.print_exc()

        if self.allow_to_generate_utg:
            self.utg.finish()
        self.logger.info("Exploration action count: %d" % self.event_count)
        self.logger.info("Exploration finished")
        bug_report_path = os.path.join(self.device.output_dir, "all_states")
        generate_report(
            bug_report_path,
            self.device.output_dir,
            self.triggered_bug_information,
            self.time_needed_to_satisfy_precondition,
            self.device.cur_event_count,
            self.time_recoder.get_time_duration(),
        )
        self.tear_down()
        
    def get_weights(self, events, counts, base_weight=1.0):
        return [base_weight * (self.decay_factor ** counts[event]) for event in events]
            
    def generate_event(self):
        if self.event_count == START_TO_GENERATE_EVENT_IN_POLICY or isinstance(
                self.last_event, ReInstallAppEvent
        ):
            self.run_initializer()
            self.from_state = self.device.get_current_state()
        current_state = self.from_state
        if current_state is None:
            time.sleep(5)
            return KeyEvent(name="BACK")

        rules_to_check = self.kea.get_rules_whose_preconditions_are_satisfied()
        for rule_to_check in rules_to_check:
            self.statistics_of_rules[str(rule_to_check.function.__name__)][
                RULE_STATE.PRECONDITION_SATISFIED
            ] += 1

        if len(rules_to_check) > 0:
            t = self.time_recoder.get_time_duration()
            self.time_needed_to_satisfy_precondition.append(t)
            self.logger.debug(
                "has rule that matches the precondition and the time duration is "
                + t
            )
            if random.random() < 0.5:
                self.logger.info("Check property")
                self.check_rule_whose_precondition_are_satisfied()
                if self.restart_app_after_check_property:
                    self.logger.debug("restart app after check property")
                    return KillAppEvent(app=self.app)
                return None
            else:
                self.logger.info("Don't check the property due to the randomness")
        
        current_state = self.from_state
        # if it is the first time to reach this state, add all possible inputs into the input table
        if current_state.state_str not in self.input_table or len(self.input_table[current_state.state_str]['events']) < 2 or random.random() < 0.5:
            possible_events = current_state.get_possible_input()
            try_cnt = 0
            while len(possible_events) == 0 and try_cnt < 5:
                time.sleep(1)
                self.from_state = self.device.get_current_state()
                current_state = self.from_state
                possible_events = current_state.get_possible_input()
                try_cnt += 1
            possible_events.append(KeyEvent(name="BACK"))
            if not self.disable_rotate:
                possible_events.append(RotateDevice())
            self._event_trace += EVENT_FLAG_EXPLORE

            self.input_table[current_state.state_str] = {}
            self.input_table[current_state.state_str]['checked_time'] = 0
            self.input_table[current_state.state_str]['events'] = []
            for event in possible_events:
                event_str = get_event_str(event, current_state)
                self.input_table[current_state.state_str]['events'].append(event_str)
                if event_str not in self.event_table:
                    self.event_table[event_str] = {
                        "event": event,
                        "checked": False,
                        "tried": 0
                    }

        if self.random_test:
            # select an event based on the input table
            counts = {}
            for event_str in self.input_table[current_state.state_str]['events']:
                # FIXME Rotate and back should be excluded?
                if event_str.startswith("RotateDevice") or event_str.startswith("KeyEvent"):
                    # 0.8 ** 7 = 0.2097152
                    counts[event_str] = 7
                else:
                    counts[event_str] = self.event_table[event_str]["tried"]
            weights = self.get_weights(self.input_table[current_state.state_str]['events'], counts)
            event_str = random.choices(self.input_table[current_state.state_str]['events'], weights=weights, k=1)[0]
            event = self.event_table[event_str]["event"]
            self.event_table[event_str]["tried"] += 1
        else:
            # use llm to select an event
            available_inputs = []
            current_page = current_state.foreground_activity.split(".")[-1]
            for event_str in self.input_table[current_state.state_str]['events']:
                if self.event_table[event_str]["tried"] < MAX_SAME_EVENT_COUNT and not self.event_table[event_str]["checked"]:
                    available_inputs.append({'event_name': event_str, 'tried': self.event_table[event_str]['tried']})
            available_inputs.sort(key=lambda x: x['tried'])

            history = self.format_action_history(3)
            has_history = False
            if history != "":
                has_history = True
                history = "\n4. [Action History]: " + history + "\n"

            checked_property_text = "\n5. The target property has already been found in the current UI state; please try exploring other screens or interface states"

            find_path_prompt = f"""Input:
1. [Target Property]: "{self.task_stack.get_current_task()}"
2. [Available Actions]: {available_inputs}
3. [UI Context]: Currently on {current_page} page {history if has_history else ""} {checked_property_text if self.input_table[current_state.state_str]['checked_time'] > 0 else ""}

Task:
Generate human-like testing decisions following:

Decision Matrix (100pts):
1. Target relevance (35pts)
  - Directly matches target properties +20
  - Indirectly related +15
2. Exploration efficiency (30pts)
  - Least-tried path +15
  - Non-redundant flow +10  
  - Progressive depth +5
3. Context coherence (25pts)
  - Continue current workflow +15
  - Avoid undo patterns (e.g. back after open) -10*
4. Innovation (10pts)

Output Requirements:
{{
  "selected_event": "event name",
  "confidence_score": 0.X,
  "reasoning": "Concise technical justification <50 chars"
}}"""
            print(find_path_prompt)
            response = query_llm("You are a senior mobile app testing strategist specializing in intelligent test path selection", find_path_prompt, self.from_state.get_state_screen())
            print(f"response: {response}")
            response_dict = extract_and_validate_json(response)
            if response_dict is None:
                self.logger.warning("LLM response is invalid")
                return None
            self.history.append(response_dict)

            selected_action = response_dict["selected_event"]
            if selected_action not in self.event_table:
                self.logger.warning("Selected action is not in the event table")
                return None

            event = self.event_table[selected_action]["event"]
            self.event_table[selected_action]["tried"] += 1

        if isinstance(event, RotateDevice):
            if self.last_rotate_events == KEY_RotateDeviceToPortraitEvent:
                self.last_rotate_events = KEY_RotateDeviceToLandscapeEvent
                event = RotateDeviceToLandscapeEvent()
            else:
                self.last_rotate_events = KEY_RotateDeviceToPortraitEvent
                event = RotateDeviceToPortraitEvent()

        return event

    def format_action_history(self, n=3):
        return ";".join(
            f"{i+1}. {a['selected_event'][:35]} ({a['reasoning'][:50]})"
            for i, a in enumerate(self.history[-n:])
        )
        
    def process_event(self, event, input_manager):
        if event is not None:
            try:
                self.device.save_screenshot_for_report(
                    event=event, current_state=self.from_state
                )
            except:
                self.from_state = self.device.get_current_state()
                self.device.save_screenshot_for_report(event=event, current_state=self.from_state)
            finally:
                input_manager.add_event(event)
                self.event_count += 1
        
        self.last_event = event
        self.to_state = self.device.get_current_state()
        if self.allow_to_generate_utg:
            self.update_utg()
            
    def update_utg(self):
        self.utg.add_transition(self.last_event, self.from_state, self.to_state, False)

    def generate_llm_event(self):
        # This method is overridden to validate the LLM response
        if self._in_llm:
            self.action_prompt3()
            self.llm()
            self.check_prompt()
            response = extract_and_validate_json(str(self.llm().content))
            if response is None:
                self.logger.warning("LLM response is invalid")
                return None
            act = Action(**response)
            self._in_llm = act.hasNext
            return U2Event(act)
        self._messages = []
        self.meaning_prompt()
        self.llm()
        self.action_prompt1()
        res = self.llm().content
        self._generated_tasks.add(res.split('\n')[0])
        self.action_prompt2()
        self.llm()
        self.check_prompt()
        response = extract_and_validate_json(str(self.llm().content))
        if response is None:
            self.logger.warning("LLM response is invalid")
            return None
        act = Action(**response)
        self._in_llm = act.hasNext
        return U2Event(act)

    def action_prompt2(self):
        # This method is overridden to support swipe actions
        prompt = """Please describe the **first step** of the operation you just performed in JSON format, as shown below:
    {
        "action": "input_text",
        "selectors": {"resourceId": "com.example:id/input", "text": "password"},
        "inputText": "123456",
        "hasNext": true,
        "direction": "left"
    }
    Notes:
    - The "action" must be one of: click, long_click, input_text, press_enter, swipe
    - "selectors" can only include: **text**, **className**, **description**, **resourceId**, and must be in camelCase. You can not use other selectors.
    - The value is the value of the selector, which must be found in the previous XML
    - "inputText" is the text to input, only present when the action is input_text
    - "hasNext" is a boolean indicating whether there is a next step. Set it to false if there is no next step
    - "direction" is the direction of the swipe, only present when the action is swipe. It can be "left", "right", "up", or "down".
    Try to combine multiple selectors to uniquely identify the element.
    Please return the operation in JSON format only. Do not explain or use code blocks.
    """
        self._messages.append({"role": "user", "content": prompt}) 
            
            
class EnhancePolicy(EnhancedNewPolicy):
    def __init__(
            self,
            device: "Device",
            app: "App",
            kea: "Kea" = None,
            restart_app_after_check_property=False,
            number_of_events_that_restart_app=100,
            clear_and_restart_app_data_after_100_events=False,
            allow_to_generate_utg=False,
            output_dir: str = None,
            decay_factor=0.8,
            disable_rotate=False,
    ):
        super(EnhancePolicy, self).__init__(device, app, kea, output_dir=output_dir,
                                        restart_app_after_check_property=restart_app_after_check_property,
                                        number_of_events_that_restart_app=number_of_events_that_restart_app,
                                        clear_and_restart_app_data_after_100_events=clear_and_restart_app_data_after_100_events,
                                        allow_to_generate_utg=allow_to_generate_utg, disable_rotate=disable_rotate,decay_factor=decay_factor)
    
    def start(self, input_manager: "InputManager"):
        self.event_count = 0
        self.input_manager = input_manager
        while input_manager.enabled and self.event_count < input_manager.event_count:
            try:
                if self.device.is_harmonyos == False and hasattr(self.device, "u2"):
                    self.device.u2.set_fastinput_ime(True)

                self.logger.info("Exploration action count: %d" % self.event_count)

                if self.to_state is not None:
                    self.from_state = self.to_state
                else:
                    self.from_state = self.device.get_current_state()
                    
                self.device.from_state = self.from_state

                if self.event_count == 0:
                    event = KillAppEvent(app=self.app)
                elif self.event_count == 1:
                    event = IntentEvent(self.app.get_start_intent())
                elif self.event_count % self.number_of_events_that_restart_app == 0:
                    if self.clear_and_reinstall_app:
                        self.logger.info(
                            "clear and reinstall app after %s events"
                            % self.number_of_events_that_restart_app
                        )
                        event = ReInstallAppEvent(self.app)
                    else:
                        self.logger.info(
                            "restart app after %s events" % self.number_of_events_that_restart_app
                        )
                        event = KillAndRestartAppEvent(app=self.app)
                    self._generated_tasks.clear()
                else:
                    self.move_if_need()
                    event = self.generate_event()
                self.process_event(event, input_manager)

            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.info("stop sending events: %s" % e)
                self.logger.info("action count: %d" % self.event_count)
                break

            except RuntimeError as e:
                self.logger.info("RuntimeError: %s, stop sending events" % e)
                break

            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback
                traceback.print_exc()

        if self.allow_to_generate_utg:
            self.utg.finish()
        self.logger.info("Exploration action count: %d" % self.event_count)
        self.logger.info("Exploration finished")
        bug_report_path = os.path.join(self.device.output_dir, "all_states")
        generate_report(
            bug_report_path,
            self.device.output_dir,
            self.triggered_bug_information,
            self.time_needed_to_satisfy_precondition,
            self.device.cur_event_count,
            self.time_recoder.get_time_duration(),
        )
        self.tear_down()
        
        