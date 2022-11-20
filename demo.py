import minedojo
from wrappers import MineClipWrapper


def run():
    env = minedojo.make(
        task_id="combat_spider_plains_leather_armors_diamond_sword_shield",
        image_size=(160, 256),
        world_seed=123,
        seed=42,
    )
    prompts = [
        "find spider",
        "kill spider",
    ]
    env = MineClipWrapper(env,prompts)
    for episode in range(2):
        env.reset()
        done = False
        while not done:
            obs, reward, done, info = env.step(env.action_space.no_op())
        env.change_prompt()
    env.close()

if __name__ == "__main__":
    run()



