import os

print(max([int(y.replace("model", "")) for y in [x.replace(".keras", "") for x in [f for f in os.listdir("./saved/") if os.path.isfile(os.path.join("./saved/", f))]]]))