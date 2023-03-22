starting_learning_rate = 1.
learning_decay = 0.1

for step in range(10000):
    learning_rate = starting_learning_rate * (1 / (1 + learning_decay * step))

    if not step % 100:
        print(learning_rate)