
import json
import random


def save_jsonl(data, file):
    with open(file, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

def path_stringfy(input_list, shuffle=True):
    output_list = []
    for i in range(len(input_list) - 1):
        output_list.append(f'{input_list[i]},{input_list[i+1]}')
    if shuffle:
        random.shuffle(output_list)
    return '/'.join(output_list)

def paths_stringfy(input_lists, shuffle=True):
    output_list = []
    for input_list in input_lists:
        for i in range(len(input_list) - 1):
            output_list.append(f'{input_list[i]},{input_list[i+1]}')
    if shuffle:
        random.shuffle(output_list)
    return '/'.join(output_list)

def convert_to_jsonline(data_list):
    res = []
    for data in data_list:
        answer_str = path_stringfy(data[0], shuffle=False)
        reverse_answer_str = path_stringfy(data[0][::-1], shuffle=False)
        path_str = paths_stringfy(data)
        input_str = path_str + f'-{data[0][0]},{data[0][-1]}'
        res.append({"input": input_str, "output": answer_str, "reversed": reverse_answer_str})
    return res

def generate_streams(num):
    data = []
    numbers = list(range(NODE_PER_STREAM*STREAM))
    while len(data) < num:
        sep_position = random.randint(0, NODE_PER_STREAM-1)
        random.shuffle(numbers)
        if str(numbers) in cache:
            continue
        else:
            cache.add(str(numbers))

        streams = [numbers[NODE_PER_STREAM*i:NODE_PER_STREAM*(i+1)] for i in range(STREAM)]
        answer = streams[0]
        for s in streams[1:]:
            s[sep_position] = answer[sep_position]
        data.append(streams)
        print(len(data))
    return data


STREAM = 2
NODE_PER_STREAM = 14
NUM_TRAIN = 1000000
NUM_TEST = 1000
SEED = 1

random.seed(SEED)
cache = set()
test_streams = generate_streams(NUM_TEST)
train_streams = generate_streams(NUM_TRAIN)

save_jsonl(convert_to_jsonline(test_streams), f"path_test-{STREAM}-{NODE_PER_STREAM}.jsonl")
save_jsonl(convert_to_jsonline(train_streams), f"path_train-{STREAM}-{NODE_PER_STREAM}.jsonl")