import pickle
with open("./adv_data_PIK.pickle","rb") as file:
    data=pickle.load(file)

# for i in range(num_processes):
#         with open(config['paths']['augmented_data_save_path'].replace('.pickle', f'adv_{i}.pickle'), "rb") as file:
#             data = pickle.load(file)
#             for key in combined_advdata:
#                 combined_advdata[key].extend(data[key])
print(data)