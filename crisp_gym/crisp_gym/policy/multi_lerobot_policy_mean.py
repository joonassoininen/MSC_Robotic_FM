"""Interface for a Policy interacting in CRISP."""

import json
import logging
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class
from typing_extensions import override

from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.policy.policy import Action, Observation, Policy, register_policy
from crisp_gym.util.lerobot_features import concatenate_state_features, numpy_obs_to_torch
from crisp_gym.util.setup_logger import setup_logging
import threading
from scipy.spatial.transform import Slerp

from typing import Any, Callable, List, Tuple

try:
    from lerobot.policies.factory import make_pre_post_processors
    USE_LEROBOT_PROCESSORS = True
    logging.info("Found lerobot pre/post processor support.")
except ImportError:
    USE_LEROBOT_PROCESSORS = False
    logging.warning("No lerobot pre/post processor support found.")


logger = logging.getLogger(__name__)

TASK_GRASP = "Pick up the yellow cube."
TASK_PRIMARY = "Pick up the yellow cube and place it on the green square"
TASK_PRIMARY_2 = "Pick up the yellow cube and place it on the green square"


@register_policy("lerobot_policy")
class LerobotPolicy(Policy):
    """A policy implementation that wraps a LeRobot policy for use in CRISP environments.

    This class runs LeRobot policy inference in a separate process and communicates with the
    environment to generate actions based on observations. It is intended for direct use in
    CRISP-based manipulation environments.
    """

    def __init__(
        self,
        pretrained_path: str,
        env: ManipulatorBaseEnv,
        #Added overrides
        #overrides: dict | None = None,
        overrides={"n_action_steps": 10, "chunk_size": 10}
    ):
        """Initialize the policy.

        Args:
            pretrained_path (str): Path to the pretrained policy model.
            env (ManipulatorBaseEnv): The environment in which the policy will be applied.
            overrides (dict | None): Optional overrides for the policy configuration.
        """
        self.parent_conn, self.child_conn = Pipe()
        self.env = env
        self.overrides = overrides if overrides is not None else {}

        self.inf_proc = Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_paths": [pretrained_path],
                "env": env,
                "overrides": self.overrides,
            },
            daemon=True,
        )
        self.inf_proc.start()

    @override
    def make_data_fn(self) -> Callable[[], Tuple[Observation, Action]]:  # noqa: ANN002, ANN003
        """Generate observation and action by communicating with the inference worker."""

        def _fn() -> tuple:
            """Function to apply the policy in the environment.

            This function observes the current state of the environment, sends the observation
            to the inference worker, receives the action, and steps the environment.

            Returns:
                tuple: A tuple containing the observation from the environment and the action taken.
            """
            logger.debug("Requesting action from policy...")
            obs_raw: Observation = self.env.get_obs()

            obs_raw["observation.state"] = concatenate_state_features(obs_raw)

            self.parent_conn.send(obs_raw)
            action: Action = self.parent_conn.recv().squeeze(0).to("cpu").numpy()
            logger.debug(f"Action: {action}")

            try:
                self.env.step(action, block=False)
            except Exception as e:
                logger.exception(f"Error during environment step: {e}")

            return obs_raw, action

        return _fn

    @override
    def reset(self):
        """Reset the policy state."""
        self.parent_conn.send("reset")

    @override
    def shutdown(self):
        """Shutdown the policy and release resources."""
        self.parent_conn.send(None)
        self.inf_proc.join()


@register_policy("multi_lerobot_policy")
class MultiLerobotPolicyMean(Policy):
    """ Class for creating the multipolicy environment"""

    def __init__(self, pretrained_paths: List[str], 
                 env: ManipulatorBaseEnv, 
                 #overrides: dict | None = None,
                 overrides={"n_action_steps": 10, "chunk_size": 10}, 
                 primary_idx: int = 0, 
                 translation_dims: slice = slice(0,3), 
                 gripper_dim: int = 6, primary_weight: 
                 float = 0.0,
            ):

        if len(pretrained_paths) < 2:
            raise ValueError("MultiLerobotPolicy requires at least 2 pretrained paths.")

        self.parent_conn, self.child_conn = Pipe()
        self.env = env
        self.overrides = overrides if overrides is not None else {}

        self.pretrained_paths = pretrained_paths
        self._active_indices = {primary_idx}
        self.primary_idx = primary_idx
        self.translation_dims = translation_dims
        self.gripper_dim = gripper_dim
        self.primary_weight = primary_weight

        self.gripping_enabled = False

        self.already_switched = False

        self.secodary_model_switched_off = False

        self.reseted_primary = True

        self.send_lock = threading.Lock()

        self.inf_proc = Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_paths": pretrained_paths,
                "env": env,
                "overrides": self.overrides,
                "primary_idx": primary_idx,
                "translation_dims": translation_dims,
                "gripper_dim": gripper_dim,
                "primary_weight": primary_weight,
                "gripping_enabled": self.gripping_enabled,
                "already_switched": self.already_switched,
                "secodary_model_switched_off": self.secodary_model_switched_off,
                "reseted_primary": self.reseted_primary,
            },
            daemon=True,
        )
        self.inf_proc.start()
    
    @property
    def active_indices(self) -> int:
        """ Returns list of active policies, 0 is primary policy and 1 is grasping policy"""
        return self._active_indices
    
    @property
    def active_path(self) -> str:
        """ Returns the primary path"""
        return self.pretrained_paths[self.primary_idx]
    
    def toggle(self, idx: int):
        """ toggles between the policies"""

        with self.send_lock:
            if idx == self.primary_idx:
                logger.warning("Primary model cannot be switched off")
                return

            if idx in self.active_indices:
                self.active_indices.discard(idx)
                logger.info(f"Model idx: {idx} switched off")
                self.reseted_primary = False
            
            else: 
                self.active_indices.add(idx)
                logger.info(f"Model idx: {idx} activated alongside primary model")
            
            self.parent_conn.send(f"toggle:{idx}")

    def start_keyboard_listener(self, switch_key: str = "m", target_idx: int = 1):
        """ By pressing "m" on the keyboard, the model toggle is performed
            
            Used mostly during the development phase
        """
        def listen():
            while True:
                key = input()
                if key.strip().lower() == "m":
                    self.toggle(target_idx)

        threading.Thread(target=listen, daemon=True).start()
        logger.info(f"[KeySwitch] Keyboard listener started. Press '{switch_key}' to switch policies.")
    
    @override
    def make_data_fn(self) -> Callable[[], Tuple[Observation, Action]]:
        return _make_data_fn(self.env, self.parent_conn, self.send_lock)
 
    @override
    def reset(self):
        self.gripping_enabled = False
        self.parent_conn.send("reset")
 
    @override
    def shutdown(self):
        self.parent_conn.send(None)
        self.inf_proc.join()

def _make_data_fn(env: ManipulatorBaseEnv, parent_conn: Connection, send_lock) -> Callable:
    """Generate observation and action by communicating with the inference worker."""

    def _fn() -> tuple:
        """Function to apply the policy in the environment.

            This function observes the current state of the environment, sends the observation
            to the inference worker, receives the action, and steps the environment.

            Returns:
                tuple: A tuple containing the observation from the environment and the action taken.
            """

        logger.debug("Requesting action from policy...")
        obs_raw: Observation = env.get_obs()
        obs_raw["observation.state"] = concatenate_state_features(obs_raw)
        with send_lock:
            parent_conn.send(obs_raw)
            action: Action = parent_conn.recv().squeeze(0).to("cpu").numpy()
        logger.debug(f"Action: {action}")
        try:
            env.step(action, block=False)
        except Exception as e:
            logger.exception(f"Error during environment step: {e}")
        return obs_raw, action

    return _fn

def load_policy(pretrained_path: str, env: ManipulatorBaseEnv, device: torch.device, overrides: dict, logger: logging.Logger):
    """ Loads policy """
    
    logger.info(f"[Inference] Loading training config from {pretrained_path}...")

    train_config = TrainPipelineConfig.from_pretrained(pretrained_path)
    _check_dataset_metadata(train_config, env, logger)
    logger.info(f"[Inference] Loaded training config")

    if train_config.policy is None:
        raise ValueError(f"Policy configuration is missing in {pretrained_path}")
    logger.info(f"[Inference] Loading policy from {pretrained_path} ... ")
    policy_cls = get_policy_class(train_config.policy.type)

    policy = policy_cls.from_pretrained(pretrained_path)

    for key, value in (overrides or {}).items():
        logger.warning(f"[Inference] Overridoverridese: {key} = {getattr(policy.config, key)} -> {value}")

        setattr(policy.config, key, value)
    policy.reset()
    policy.to(device).eval()

    preprocessor, postprocessor = None, None
    if USE_LEROBOT_PROCESSORS:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=pretrained_path
        )
    logger.info(f"[Inference] Loaded {policy.name} from {pretrained_path} on {device}.")
    return policy, preprocessor, postprocessor

def _make_warmup_obs(policy, env: ManipulatorBaseEnv, device: torch.device) -> dict:

    warmup_obs_raw = env.observation_space.sample()
    warmup_obs_raw["observation.state"] = concatenate_state_features(warmup_obs_raw)

    warmup_obs = numpy_obs_to_torch(warmup_obs_raw)
 
    return warmup_obs

def _warmup_policy(policy, preprocessor, warmup_obs: dict, logger: logging.Logger, n_runs: int = 100):

    import time
    elapsed_list = []

    if USE_LEROBOT_PROCESSORS:
        warmup_obs = preprocessor(warmup_obs)

    with torch.inference_mode():
        for _ in range(100):
            start = time.time()
            _ = policy.select_action(warmup_obs)
            end = time.time()
            elapsed = end - start
            elapsed_list.append(elapsed)
    avg_elapsed = sum(elapsed_list) / len(elapsed_list)
    std_elapsed = np.std(elapsed_list)
    max_elapsed = max(elapsed_list)
    min_elapsed = min(elapsed_list)
    logger.info(
        f"[Inference] Warm-up timing over 100 runs: "
        f"avg={avg_elapsed * 1000:.2f}ms, std={std_elapsed * 1000:.2f}ms, max={max_elapsed * 1000:.2f}ms, min={min_elapsed * 1000:.2f}ms"
    )


def inference_worker(
    conn: Connection,
    pretrained_paths: List[str],
    env: ManipulatorBaseEnv,
    #overrides: dict | None = None,
    overrides={"n_action_steps": 10, "chunk_size": 10},
    primary_idx: int = 0,
    translation_dims: slice = slice(0, 3),
    primary_weight: float = 0.0,
    gripper_dim: int = 6,
    gripping_enabled: bool = False,
    already_switched: bool = False,
    secodary_model_switched_off = False,
    primary_already_reseted = False,
    reseted_primary = True,
    place_task_active = False,
):

    """
        conn: connection between lerobot model and crisp
        pretrained_paths: List of loaded policies
        env: Selected CRISP environment
        overrides: Overrides default values from the policy configuration 
        primary_idx: Index of the primary policy
        translation_dims: translation dimensions from the produced actions
        primary_weight: How much primary actions effects to the produced action,
        set to 0.0 but some tests included some weighted average caluclations from both policy
        outputs. 
        gripper_dim: Gripper action dimension from the produced action
        gripping_enabled: Has the grasping policy allowed the gripper to close
        already_switched: Has the grasping policy already been switched. Relates to the policy switching process
        secodary_model_switched_off: Additional variable that relates to the policy switching process
        primary_already_reseted: Is the primary policy reseted after succesful grasp. Relates to the policy switching process
        reseted_primary: Additional variable that relates to the policy switching process
        place_task_active:
    """

    setup_logging()
    logger = logging.getLogger(__name__)

    try: 
        from lerobot.utils.import_utils import register_third_party_plugins
        register_third_party_plugins()
    except ImportError:
        logger.warning(
            "[Inference] Could not import third-party plugins for LeRobot. Continuing without them.")
    logger.info("[Inference] Starting inference worker...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Inference] Using device: {device}")

        policies = []

        for path in pretrained_paths:
            policy = load_policy(path, env, device, overrides or {}, logger)
            policies.append(policy)
        
        active_indices= {primary_idx}
        logger.info(f"[Inference] All {len(policies)} model(s) loaded. Active: index {active_indices}.")

        for i, (policy, preprocessor, postprocessor) in enumerate(policies):

            if reseted_primary == False:
                policy, preprocessor, postprocessor = policies[0]
                policy.reset()
                if USE_LEROBOT_PROCESSORS:
                    preprocessor.reset()
                    postprocessor.reset()
                print("PRIMARY POLICY RESETED")
                reseted_primary = True

            logger.info(f"[Inference] Warming up policy {i}: {pretrained_paths[i]}...")
            warmup_obs = _make_warmup_obs(policy, env, device)

            if i == 1:
                """ Removes the z robot state observation from the grasping policy"""
                state = warmup_obs["observation.state.cartesian"]
                state1 = warmup_obs["observation.state"]

                keep_idx = [j for j in range(state.shape[-1]) if j != 2]
                keep_idx1 = [k for k in range(state1.shape[-1]) if k != 2]

                warmup_obs["observation.state.cartesian"] = state[..., keep_idx]
                warmup_obs["observation.state"] = state1[..., keep_idx1]
                

            
            _warmup_policy(policy, preprocessor, warmup_obs, logger, n_runs=100)
            policy.reset()
            if USE_LEROBOT_PROCESSORS:
                preprocessor.reset()
                postprocessor.reset()
        logger.info("[Inference] All warm-ups complete. Entering inference loop...")

        while True: 
            logger.debug("[Inference] Waiting for message...")

            msg = conn.recv()

            if msg is None:
                logger.info("[Inference] Shutdown signal received.")

                break
            if msg == "reset":
                logger.info(f"[Inference] Resetting policy {active_indices}.")
                for idx in active_indices:
                    policy, preprocessor, postprocessor = policies[idx]
                    policy.reset()
                    gripping_enabled = False
                    already_switched = False
                    secodary_model_switched_off = False
                    primary_already_reseted = False
                    place_task_active = False
                    if USE_LEROBOT_PROCESSORS:
                        preprocessor.reset()
                        postprocessor.reset()
                continue
            
            if isinstance(msg, str) and msg.startswith("toggle:"):
                """
                If toggle message is sent to the connection, the grasping policy is either switched
                on or switched off
                """
                idx = int(msg.split(":")[1])
                if idx < 0 or idx >= len(policies):
                    logger.warning(f"[Inference] Invalid switch index {idx}, ignoring.")
                    continue

                if idx == primary_idx:
                    logger.warning(f"[Inference] Cannot switch off the primary model, ingoring")
                    continue
                
                if idx in active_indices:
                    active_indices.discard(idx)
                    primary_policy, primary_preprocessor, primary_postprocessor = policies[primary_idx]
                    primary_policy.reset()
                    if USE_LEROBOT_PROCESSORS:             
                        primary_preprocessor.reset()             
                        primary_postprocessor.reset()
                        print("processors reseted !!")
                    print("PRIMARY RESETED !!!")
                    place_task_active = True
                    reseted_primary = False
                    gripping_enabled = False
                    logger.info(f"[Inference] Policy {idx} deactivated. Active: {active_indices}.")
                
                else:
                    active_indices.add(idx)
                    policy, preprocessor, postprocessor = policies[idx]
                    policy.reset()
                    if USE_LEROBOT_PROCESSORS:
                        preprocessor.reset()
                        postprocessor.reset()
                    logger.info(f"[Inference] Policy {idx} activated alongside {primary_idx}. Active: {active_indices}.")
                continue

            obs_raw = msg
            with torch.inference_mode():
                obs = numpy_obs_to_torch(obs_raw)
                actions = {}

                #z = obs["observation.state.cartesian"][0, 2].item()
                #print(z)
 
                if already_switched == False and secodary_model_switched_off == False:
                    """ When the end-effector height goes under 0.17 for the first time the grasping policy is activated"""
                    if z < 0.17: 
                        active_indices.add(1)
                        policy, preprocessor, postprocessor = policies[1]
                        policy.reset()
                        
                        if USE_LEROBOT_PROCESSORS:
                            preprocessor.reset()
                            postprocessor.reset()
                        already_switched = True                        

                for idx in active_indices:
                    policy, preprocessor, postprocessor = policies[idx]

                    if idx == 0 and place_task_active == True:
                        obs = {**obs, "task": TASK_PRIMARY_2}

                    if idx == 1:
                         """ Removes the z robot state observation from the grasping policy"""
                        state = obs["observation.state.cartesian"]
                        state1 = obs["observation.state"]
                        keep_idx = [j for j in range(state.shape[-1]) if j != 2]
                        keep_idx1 = [k for k in range(state1.shape[-1]) if k != 2]
                        obs = {**obs, "observation.state.cartesian": state[..., keep_idx]}
                        obs = {**obs, "observation.state": state1[..., keep_idx1]}
                        #obs = {**obs, "task": TASK_GRASP}

                    obs_i = preprocessor(obs) if USE_LEROBOT_PROCESSORS else obs

                    action_i = policy.select_action(obs_i)
                    # Stores both actions into action storage
                    if USE_LEROBOT_PROCESSORS:
                        action_i = postprocessor(action_i)
                    actions[idx] = action_i

                if len(actions) == 1:
                    action = actions[primary_idx]
                else:
                    secondary_idx = next(idx for idx in active_indices if idx != primary_idx)

                    primary_action = actions[primary_idx]
                    secondary_action = actions[secondary_idx]

                    action = primary_action.clone()

                    # Calculates the final action
                    # If weighted average is used by modifying the primary weight variable
                    # The fnal action is not completely the final action 
                    action[..., translation_dims] = (
                        primary_weight * primary_action[..., translation_dims] 
                        + (1 - primary_weight) * secondary_action[..., translation_dims]
                    )

                    #print("Secondary model z action:", secondary_action[..., 2])
                    
                    # Allows the primary policy to grasp
                    if secondary_action[..., gripper_dim] < 0.91:
                        gripping_enabled = True
                    
                    # After the grasping is enabled, the primary policy can control the gripper
                    if gripping_enabled == True: 
                        action[..., gripper_dim] = primary_action[..., gripper_dim]
                    else:
                        action[..., gripper_dim] = 1.0

                    #print("gripper: ", action[..., gripper_dim])

                    if secodary_model_switched_off == False and gripping_enabled == True:
                        #if primary_action[..., gripper_dim] > 0.5:
                        if len(active_indices) > 1:
                            
                            if z > 0.068 and primary_already_reseted == False:
                                # When the object is lifted, primary policy is reset before continuing the task
                                primary_policy, primary_pre, primary_post = policies[0]
                                primary_policy.reset()
                                if USE_LEROBOT_PROCESSORS:
                                    primary_pre.reset()
                                    primary_post.reset()
                                primary_already_reseted = True
                                print("Primary Reseted !!")
                                active_indices.discard(1)
                                    
                                gripping_enabled = False
                                secodary_model_switched_off = True
                    
                    #action[..., gripper_dim] = primary_action[..., gripper_dim]

            logger.debug(f"[Inference] Action from policy {active_indices}: {action}")
            conn.send(action)
    
    
    except Exception as e:
        logger.exception(f"[Inference] Exception in inference worker: {e}")
 
    conn.close()
    logger.info("[Inference] Worker shutting down.")

def _check_dataset_metadata(
    train_config: TrainPipelineConfig,
    env: ManipulatorBaseEnv,
    logger: logging.Logger,
    keys_to_skip: list[str] | None = None,
):
    """Check if the dataset metadata matches the environment configuration.

    Args:
        train_config (TrainPipelineConfig): The training pipeline configuration.
        env (ManipulatorBaseEnv): The environment to compare against.
        logger (logging.Logger): Logger for logging information.
        keys_to_skip (list[str] | None): List of metadata keys to skip during comparison.
    """
    if keys_to_skip is None:
        keys_to_skip = []

    def _warn_if_not_equal(key: str, env_val: Any, policy_val: Any):
        if env_val != policy_val:
            logger.warning(
                f"[Inference] Mismatch in metadata for key '{key}': "
                f"env has '{env_val}', policy has '{policy_val}'."
            )

    def _warn_if_missing(key: str):
        logger.warning(f"[Inference] Key '{key}' not found in environment metadata.")

    try:
        metadata = LeRobotDatasetMetadata(repo_id=train_config.dataset.repo_id)
        logger.debug(f"[Inference] Loaded dataset metadata: {metadata}")

        path_to_metadata = Path(metadata.root / "meta" / "crisp_meta.json")
        if path_to_metadata.exists():
            logger.info(
                "[Inference] Found crisp_meta.json in dataset, comparing environment and policy configs..."
            )
            env_metadata = env.get_metadata()
            with open(path_to_metadata, "r") as f:
                dataset_metadata = json.load(f)
            for key, value in dataset_metadata.items():
                if key in keys_to_skip:
                    continue
                if isinstance(value, dict):
                    if key not in env_metadata:
                        _warn_if_missing(key)
                        continue
                    for subkey, subvalue in value.items():
                        if subkey not in env_metadata[key]:
                            _warn_if_missing(f"{key}.{subkey}")
                            continue
                        _warn_if_not_equal(
                            f"{key}.{subkey}",
                            env_metadata[key].get(subkey),
                            subvalue,
                        )
                else:
                    if key not in env_metadata:
                        _warn_if_missing(key)
                    _warn_if_not_equal(key, env_metadata.get(key), value)

    except Exception as e:
        logger.warning(f"[Inference] Could not load dataset metadata: {e}")
        logger.info("[Inference] Skipping metadata comparison.")