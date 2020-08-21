from gym import envs
env_names = [spec.id for spec in envs.registry.all()]
i = 0
for name in sorted(env_names):
    print(name)
    i+=1
print(i)