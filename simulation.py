import random
snr = 10
error = random.random()
print("communication simulation start")
print("now SNR:",snr)
if error < 0.5:
    print("receive suucessful")
else:
    print("error")
