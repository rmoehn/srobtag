from gym.envs.registration import register

# Credits: https://github.com/openai/gym/tree/master/gym/envs

register(
    id='srobtag-v0',
    entry_point='gym_srobtag.envs:SrobtagEnv',
)
