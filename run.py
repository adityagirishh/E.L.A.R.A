from environment import ElaraEnv
from models import Action

env = ElaraEnv()
obs = env.reset()
print(obs.model_dump())

obs, reward, done, info = env.step(
    Action(
        action_type="email",
        target_lead_id="L-001",
        subject="Requesting your documents",
        content="Hi Arun, sharing the documents request as discussed.",
    )
)

print(obs.model_dump())
print(reward, done, info)