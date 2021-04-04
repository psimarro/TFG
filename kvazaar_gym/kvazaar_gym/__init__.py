from gym.envs.registration import register

register(
    id='kvazaar-v0',
    entry_point='kvazaar_gym.envs:Kvazaar',
)