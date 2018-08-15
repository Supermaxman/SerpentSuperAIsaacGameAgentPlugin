import time
from datetime import datetime

from serpent.enums import InputControlTypes
from serpent.frame_grabber import FrameGrabber
from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey

from .super_ml import super_agent

class SerpentSuperAIsaacGameAgent(GameAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):
        Bosses = self.game.environment_data["BOSSES"]
        #Items = self.game.environment_data["ITEMS"]

        self.environment = self.game.environments["BOSS_FIGHT"](
            game_api=self.game.api,
            input_controller=self.input_controller,
            bosses=[
                Bosses.MONSTRO
            ],
            items={
                Bosses.MONSTRO: []
            }
        )

        self.game_inputs = [
            {
                "name": "MOVEMENT",
                "control_type": InputControlTypes.DISCRETE,
                "inputs": self.game.api.combine_game_inputs(["MOVEMENT"])
            },
            {
                "name": "SHOOTING",
                "control_type": InputControlTypes.DISCRETE,
                "inputs": self.game.api.combine_game_inputs(["SHOOTING"])
            },
        ]

        self.agent = super_agent.SuperAIsaacAgent(
            "SuperAIsaacPPO-V29",
            game_inputs=self.game_inputs,
            callbacks=dict(
                after_observe=self.after_agent_observe,
                before_update=self.before_agent_update,
                after_update=self.after_agent_update
            ),
            seed=0
        )
        self.agent.build()
        try:
            print('Loading model...')
            self.agent.load()
            print('Loaded model from checkpoint.')
        except FileNotFoundError:
            print('Initializing new model...')
            self.agent.init()
            print('Initialized model.')
        self.agent.count_trainable_variables()
        self.started_at = datetime.utcnow().isoformat()

        self.analytics_client.track(event_key="GAME_NAME", data={"name": "The Binding of Isaac: Afterbirth+"})

        self.environment.new_episode(maximum_steps=3840)

    def handle_play(self, game_frame, game_frame_pipeline):
        valid_game_state = self.environment.update_game_state(game_frame)

        if not valid_game_state:
            return None

        move_reward, attack_reward = self.reward_aisaac(self.environment.game_state, game_frame)

        terminal = (
            not self.environment.game_state["isaac_alive"] or
            self.environment.game_state["boss_dead"] or
            self.environment.episode_over
        )

        self.agent.observe(
            move_reward=move_reward, 
            attack_reward=attack_reward, 
            terminal=terminal, 
            boss_hp=self.environment.game_state["boss_hp"], 
            isaac_hp=self.environment.game_state["isaac_hp"])

        if not terminal:
            #[0, 2, 4, 6] to [0, 2]
            # 30fps, look at current frame and frame from 1/30 s/f * 4 = 0.13 seconds ago
            frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")
            agent_actions = self.agent.generate_actions(frame_buffer)
            #print(agent_actions)
            self.environment.perform_input(agent_actions)
        else:
            self.environment.clear_input()

            self.agent.reset()

            if self.environment.game_state["boss_dead"]:
                self.analytics_client.track(event_key="BOSS_KILL", data={"foo": "bar"})

            self.environment.end_episode()
            self.environment.new_episode(maximum_steps=3840, reset=False)

    def reward_aisaac(self, game_state, game_frame):
        move_reward = 0.0
        attack_reward = 0.0
        if game_state["isaac_alive"]:
            if game_state["damage_taken"]:
                damage_taken = game_state["isaac_hps"][1] - game_state["isaac_hps"][0]
                move_reward += -1.0 * damage_taken
            else:
                if game_state["damage_dealt"]:
                    move_reward += 0.05
                    attack_reward += 0.05
                else:
                    move_reward += 0.01
            if game_state["boss_dead"]:
                self.agent.total_wins += 1
                move_reward += 1.0
                attack_reward += 1.0
        else:
            move_reward += -2.0
        return move_reward, attack_reward
    # Callbacks

    def after_agent_observe(self):
        self.environment.episode_step()

    def before_agent_update(self):
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)

    def after_agent_update(self):
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
        time.sleep(1)
