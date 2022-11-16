import minedojo
from wrappers import MinedojoWrapper


def run():
    env = minedojo.make(
        task_id="combat_spider_plains_leather_armors_diamond_sword_shield",
        image_size=(288, 512),
        world_seed=123,
        seed=42,
    )
    prompts = [
        "kill spider",
    ]
    env = MinedojoWrapper(env,prompts)
    env.reset()
    for _ in range(1):
        obs, reward, done, info = env.step(env.action_space.no_op())
        print(reward)
    env.close()

if __name__ == "__main__":
    run()



