import random

dropout_rate = 0.5

example_output = []
for i in range(5):
    example_output.append(i*2)

while True:

    index = random.randint(0, len(example_output) - 1)
    example_output[index] = 0;

    dropped_out = 0
    for value in example_output:
        if value ==0:
            dropped_out +=1

    if dropped_out / len(example_output) >= dropout_rate:
        break

print(example_output)