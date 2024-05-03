from asyncio import sleep
from asyncio.windows_events import NULL
from ctypes.wintypes import BOOL
import enum
from functools import total_ordering
import glob
from multiprocessing import Value
import os

from numpy.random import f, rand; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # This prevents TensorFlow from generating warnings in the console
from recommender_thread import recommender_thread
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.constraints import MinMaxNorm
from keras.models import load_model
import csv
from threading import Thread, Lock
from numpy import float32, save
from numpy import load
import tkinter as tk
from tkinter import TRUE, filedialog
from tqdm import tqdm

ratings = np.ones(209171)
mutex = Lock()
t_order = 0

def main():
    global ratings
    o = input("(1) generate Rankings. (2) Train Model based on ratings (3) Combine Rating files (4) Predict (5) gen avgs (6) Import avg into ratings (7) remove movies (8) run from loading ratings: ")
    match int(o):
        case 1:#gets all ratings from the ratings.csv and turns a given range into a np.array
            get_data_from_csv()
        case 2:#
            load_ratings()
            print(ratings.shape)
            train_model()
            predict()
        case 3:
            combine_ratings()
        case 4:
            predict()
        case 5:
            avgs, std = get_avg_for_each_movie()
            #save Averages
            print("Where do you want to save a copy of the Average's Calculations?"); path = get_npy_path();save(path,avgs);print(f"Saved Average Calculations to {path}");#print(avgs)
            #save Standard Deviation
            print("Where do you want to save a copy of the Standard Deviation Calculations?"); path = get_npy_path(); save(path,std); print(f"Saved Standard Deviations to Calculations to {path}"); #print(std)
            
            import_avg_into_ratings()
            print("Where do you want to save the new ratings?")
            path = get_npy_path()
            save(path,ratings)
        case 6:
            load_ratings()
            import_avg_into_ratings()
            print("Where do you want to save the new ratings?")
            path = get_npy_path()
            save(path,ratings)
        case 7:
            remove_unused_movies()
            print("Where do you want to save the new ratings?")
            path = get_npy_path()
            save(path,ratings)
        case 8:
            remove_unused_movies()
            print("Where would you like to Save Ratings with deleted movies?")
            path = save_ratings()
            avgs, std = get_avg_for_each_movie(path)
            #save Averages
            print("Where do you want to save a copy of the Average's Calculations?"); avg_path = get_npy_path();save(avg_path,avgs);print(f"Saved Average Calculations to {avg_path}");#print(avgs)
            #save Standard Deviation
            print("Where do you want to save a copy of the Standard Deviation Calculations?"); std_path = get_npy_path(); save(std_path,std); print(f"Saved Standard Deviations to Calculations to {std_path}"); #print(std)
            #IMPORT AVGS INTO RATINGS AND SAVE
            import_avg_into_ratings(avg_path,std_path)
            print("Where do you want to save the New Ratings filled with Average Calculations?"); path = get_npy_path();save(path,ratings)
            #Train model
            train_model()
            predict()
        case 9:
            print("import Ratings with only needed movies")
            load_ratings()
            normal_dist_into_ratings()
            print("Where do you want to save the New Ratings filled with Normal distribution Calculations?"); path = get_npy_path();save(path,ratings)
            train_model()
            predict()
        case 10:
            print("Initiating Full Run")
            get_data_from_csv()
            remove_unused_movies()
            print("Where do you want to save the new ratings?")
            path = get_npy_path()
            save(path,ratings)
            avgs, std = get_avg_for_each_movie()
            #save Averages
            print("Where do you want to save a copy of the Average's Calculations?"); path = get_npy_path();save(path,avgs);print(f"Saved Average Calculations to {path}");#print(avgs)
            #save Standard Deviation
            print("Where do you want to save a copy of the Standard Deviation Calculations?"); path = get_npy_path(); save(path,std); print(f"Saved Standard Deviations to Calculations to {path}"); #print(std)

            #0's
            print("import Ratings with only needed movies")
            load_ratings()
            train_model()
            predict()
            #AVG
            print("import Ratings with only needed movies")
            load_ratings()
            
            #normal Dist
            print("import Ratings with only needed movies")
            load_ratings()
            normal_dist_into_ratings()
            print("Where do you want to save the New Ratings filled with Normal distribution Calculations?"); path = get_npy_path();save(path,ratings)
            train_model()
            predict()
              
            
def load_ratings():
    global ratings
    print("Please provide ratings file:")
    path = get_npy_path()
    ratings = np.load(path)
    
def save_ratings():
    global ratings
    path = get_npy_path()
    save(path,ratings)
    return path    
        
def remove_unused_movies():
    global ratings
    
    # Read the movie IDs from the CSV file
    print("Please provide movies.csv")
    with open(get_csv_path(), newline='') as csvfile:
        csvFile = list(csv.reader(csvfile))
        # Extract movie IDs (assuming movie IDs are in the first column)
        movie_ids = [int(row[0]) for row in csvFile[1:]]  # Skip the header row
        
    # Load the ratings file
    print("Please provide raw ratings file:")
    path = get_npy_path()
    ratings = np.load(path)
    print(f"Saved to {path}")
    
    # Initialize a list to store the indices of columns to delete
    cols_to_delete = []
    for i in range(ratings.shape[1]):
        movie_id = i + 1  # Assuming movie IDs start from 1
        if movie_id not in movie_ids:
            cols_to_delete.append(i)
    #print(cols_to_delete)
    
    # Delete columns from the ratings array
    ratings = np.delete(ratings, cols_to_delete, axis=1)   
    
def get_data_from_csv():
    print("Beginning to Get Data from CSVs")
    global ratings
    global t_order
    users_to_train_from=int(input("# to train from: "))
    starting_user=int(input("StartingUser (1-162541): "))
    ending_user = starting_user+users_to_train_from
    users_per_section = users_to_train_from/8
    sections=[[0]*2]*8
    print("Please provide ratings.csv")
    with open(get_csv_path(), newline='') as csvfile:
        csvFile = list(csv.reader(csvfile)) # Write ratings.csv to an array        
        csvFile.remove(csvFile[0])
        u=users_per_section*1
        section=0
        starting_line=0
        end_user_for_curr_section = (users_per_section*1)+starting_user
        # for index, line in enumerate(csvFile):
        for index, line in enumerate(tqdm(csvFile, desc=f'Getting Sections', unit=' lines')):
            if int(line[0]) < starting_user:
                starting_line=index+1
                continue
            if int(line[0]) <= end_user_for_curr_section:
                continue
            else:
                sections[section] = [starting_line,index-1]
                starting_line=index
                section+=1
                end_user_for_curr_section=(users_per_section*(section+1))+starting_user
                if int(line[0])>ending_user:
                    break
    threads=[0]*8
    t=0
    for index, bounds in enumerate(sections):
        print(f"Section{index}[{bounds[0]}:{bounds[1]}]")
        nt = recommender_thread(target=get_ratings, args=(csvFile[bounds[0]:bounds[1]],mutex,t))
        threads[t]=nt
        t+=1
    for t in threads:
        t.start()
    for index, t in enumerate(threads):
        t.join()
    t_order=0
    ratings=ratings[1:]
    save(f'raw_ratings-S{starting_user}-E{ending_user}.npy',ratings)

def get_avg_for_each_movie(ratings_path=NULL):
    print("Beginning to calculate Average rating for each Movie...")
    global ratings
    if ratings_path is NULL:
        print("Please provide ratings file:")
        path = get_npy_path()
        ratings = np.load(path)
    
    ratings_by_movie=ratings.transpose()
    movie_rating_avg = np.zeros(len(ratings[0]))
    movie_rating_standard_dev = np.zeros(len(ratings[0]))
        
    # Define the number of threads
    num_threads = 8
    threads = []
    
    # Split the ratings array into chunks for each thread
    chunk_size = len(ratings_by_movie) // num_threads
    chunks = np.array_split(ratings_by_movie, num_threads)
    # Define a function to calculate average for each chunk
    def calculate_avg(chunk_start,chunk,movie_rating_avg,movie_rating_standard_dev,thread):
        for index, movie in enumerate(tqdm(chunk, desc=f'Thread {thread}', unit=' movies')):
            non_zero= np.empty(shape=(0,))
            for user in movie:
                if user != 0:
                    non_zero = np.append(non_zero,user)
            if len(non_zero)==0:
                non_zero = np.append(non_zero,2.69)
            mutex.acquire()
            movie_rating_avg[index+chunk_start]=np.average(non_zero)
            movie_rating_standard_dev[index+chunk_start]=np.std(non_zero)
            mutex.release()
            #print(f"({thread}). Movie {index+chunk_start}, avg:{movie_rating_avg[index+chunk_start]}, std:{movie_rating_standard_dev[index+chunk_start]}")
    
    # Create and start threads
    for index, chunk in enumerate(chunks):
        t = Thread(target=calculate_avg, args=(index*len(chunk),chunk,movie_rating_avg,movie_rating_standard_dev,index))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    return movie_rating_avg, movie_rating_standard_dev

def import_avg_into_ratings(avg_path = NULL, std_path=NULL):
    global ratings
    print("Taking Average ratings and importing them into ratings to fill 0's")
    if avg_path is NULL:
        print("import Path to AVG's File: "); path = get_npy_path(); avgs = np.load(path); print(f"Avg's Shape: {avgs.shape}. Ratings Shape: {ratings.shape}")
    if std_path is NULL:
        print("import Path to STD's File: "); path = get_npy_path(); STDs = np.load(path); print(f"STD's Shape: {STDs.shape}. Ratings Shape: {ratings.shape}")
    
    def update_user_ratings(start, end,thread):
        global ratings
        for i, index_users in enumerate(tqdm(range(start, end), desc=f'Thread {thread}: ', unit=' users')):
            users = ratings[index_users]
            for index, movie in enumerate(users):
                if movie == 0:
                    ratings[index_users][index] = avgs[index]
    
    # Calculate the number of users per thread
    num_users = len(ratings)
    users_per_thread = num_users // 8
    
    # Create and start threads
    threads = []
    for i in range(8):
        start = i * users_per_thread
        end = start + users_per_thread if i < 7 else num_users
        t = Thread(target=update_user_ratings, args=(start, end,i))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()

    print(ratings.shape)

def normal_dist_into_ratings(avg_path = NULL, std_path=NULL):
    global ratings
    print("Taking Average ratings and importing them into ratings to fill 0's")
    if avg_path is NULL:
        print("import Path to AVG's File: "); path = get_npy_path(); avgs = np.load(path); print(f"Avg's Shape: {avgs.shape}. Ratings Shape: {ratings.shape}")
    else: avgs = np.load(avg_path)
    if std_path is NULL:
        print("import Path to STD's File: "); path = get_npy_path(); STDs = np.load(path); print(f"STD's Shape: {STDs.shape}. Ratings Shape: {ratings.shape}")
    else: STDs = np.load(std_path)
    
    def update_user_ratings(start, end,thread):
        global ratings
        for i, index_users in enumerate(tqdm(range(start, end), desc=f'Thread {thread}: ', unit=' users')):
            users = ratings[index_users]
            for index, movie in enumerate(users):
                if movie == 0:
                    random_rating = np.random.normal(avgs[index], STDs[index])
                    ratings[index_users][index] = random_rating
    
    # Calculate the number of users per thread
    num_users = len(ratings)
    users_per_thread = num_users // 8
    
    # Create and start threads
    threads = []
    for i in range(8):
        start = i * users_per_thread
        end = start + users_per_thread if i < 7 else num_users
        t = Thread(target=update_user_ratings, args=(start, end,i))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()

    print(ratings.shape)
                
    
def get_ratings(section,mutex,thread_name):
    global t_order
    print("Reformating ratings")
    global ratings
    tmp_ratings = np.ones(209171)
    u_current_user = int(section[0][0])
    #more 0's than anything else
    #0=they haven't seen/ranked the movie
    u_current_array = np.zeros(209171)
    for index, line in enumerate(tqdm(section, desc=f'Thread {thread_name}', unit=' users')):
    #for line in section:
        line_complete = False
        while not line_complete:
            #lines[0] = userID
            if int(line[0]) == u_current_user:
                #lines[1] = movieID # Lines[2] = Ranking
                u_current_array[int(line[1]) - 1] = float(line[2])
                line_complete=True
            else:
                #needs to go to before u_current_user>500
                #semaphores for rankings?
                #print(f"Thread: {thread_name} Writing for {u_current_user}!")
                tmp_ratings = np.vstack([tmp_ratings, u_current_array])
                u_current_array = np.zeros(209171)
                u_current_user = u_current_user + 1
    tmp_ratings=tmp_ratings[1:]
    while t_order != thread_name:
        ...
    mutex.acquire()
    ratings = np.vstack([ratings, tmp_ratings])
    t_order+=1
    mutex.release()
    print(f"Thread: {thread_name} Complete")

def combine_ratings():
    print("combining ratings")
    global ratings
    file1 = get_npy_path()
    f1 = get_start_and_end(file1)
    file2 = get_npy_path()
    f2 = get_start_and_end(file2)
    if f1[1] <= f2[0] or f2[1] <= f1[0]:
        print("combining files")
        ratings = np.vstack([load(file1), load(file2)])
        if f1[0] < f2[0]: save(f'raw_ratings-S{f1[0]}-E{f2[1]}.npy',ratings);#writeResults(ratings,f"raw_ratings-S{f1[0]}-E{f2[1]}.csv")
        else: save(f'raw_ratings-S{f2[0]}-E{f1[1]}.npy',ratings);#writeResults(ratings,f"raw_ratings-S{f2[0]}-E{f1[1]}.csv")
    else: print("Files intersect: Exiting"); exit(3)
    
def get_start_and_end(s):
    s_split= s.split(".")
    s_split= s_split[0].split('/')
    s_split= s_split[len(s_split)-1].split("-")
    start = int(s_split[1][1:])
    print(f"start: {start}")
    end =int(s_split[2][1:])
    print(f"end: {end}")
    return [start,end]

def train_model():
    print("Training Model")
    global ratings
    num_users, num_movies = ratings.shape
    print(ratings.shape)
    print(f"Prediction based off of {num_users} users ratings on {num_movies} movies")
    
    input_layer = Input(shape=(num_movies,))
    hidden_layer = Dense(64, activation='relu')(input_layer)
    output_layer = Dense(num_movies, activation='linear', kernel_constraint=MinMaxNorm(min_value=0.5, max_value=5))(hidden_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')  # Using mean squared error as loss
    
    # Pass the mask along with ratings during training
    #mask = mask.reshape(-1, num_movies)
    model.fit(ratings, ratings, epochs=30, batch_size=8)
    
    print("saving prediction model")
    path=get_model_path()
    model.save(path)
    print(f"prediction model saved to {path}")
    

def predict():
    global ratings
    print("starting Predictions")
    print("Please provide the prediction model")
    model_path = get_model_path()
    model = load_model(model_path)
    total_values = 62423
    new_user = np.zeros(total_values)
    
    user_name=NULL
    option=input("Do you already have a user you want to select for? (1) yes (anything else) no")
    if int(option) == 1:
        print("Please provide the user's info")
        user_path=get_npy_path()
        new_user = np.load(user_path)
        user_name=user_path.split('+')
        user_name = user_name[1]
    else:
        non_zero_ratio=0.005
    
        # Calculate how many values should be non-zero
        num_non_zero = int(non_zero_ratio * total_values)
    
        # Generate random indices to place non-zero values
        indices = np.random.choice(total_values, num_non_zero, replace=False)
    
        # Randomly assign non-zero values between 0.5 and 5 (rounded to nearest 0.5)
        values = np.random.choice(np.arange(1, 19) * 0.5, num_non_zero) * 0.5
        for index,v in enumerate(values): 
            if v < 0.5:
                values[index]=0
            elif v >5:
                values[index]=5
    
        # Assign values to the array
        new_user[indices] = values
        
        # Reshape the array
        new_user = new_user.reshape(1, -1)
        
        user_name=input("enter the users name: ")
        user_name = user_name.replace(" ","")
        save(f'user+{user_name}+input.npy',new_user)
        
    print(f"Generating Predictions for {user_name}")
    
    prediction = model.predict(new_user)
    prediction = np.array(prediction)
    prediction = np.ravel(prediction)
    prediction = scale_values(prediction)

    # prediction = np.maximum(prediction, 0.5)
    # prediction = np.minimum(prediction, 5)
    #       we shouldn't be getting negative values in the predictions
    
    print("Please Provide Movie IDs")
    with open(get_csv_path(), newline='') as csvfile:
        csvFile = list(csv.reader(csvfile))
        # Extract movie IDs (assuming movie IDs are in the first column)
        movie_ids = [int(row[0]) for row in csvFile[1:]]
    movie_ids = np.array(movie_ids)
    movie_ids = np.transpose(movie_ids)
    ####PRINT$$$$
    new_user = np.transpose(new_user)
    prediction = np.ravel(prediction)
    # for index,v in enumerate(prediction): 
    #         if v < 1:
    #             prediction[index]=0
    #         elif v >5:
    #             prediction[index]=5
    p2=prediction
    prediction = np.transpose(prediction)
    
    # Combine random values with predictions into a single array
    for index, i in enumerate(new_user):
        if i == 0:
            new_user[index]=NULL
    data = np.column_stack((movie_ids,new_user, prediction))

    # Specify the file path where you want to save the CSV file
    prediction_model_name=model_path.split('.')
    prediction_model_name=prediction_model_name[0].split('/')
    prediction_model_name = prediction_model_name[len(prediction_model_name)-1]
    prediction_full_path = f"{user_name}_Predictions_Full_{prediction_model_name}.csv"
    prediction_path = f"{user_name}_Predictions_{prediction_model_name}.csv"
    # Save the combined data to a CSV file with headers
    header = "Movie Ids, New User Initial Ratings, Predictions"
    np.savetxt(prediction_full_path, data, delimiter=",", header=header, comments='')
    np.savetxt(prediction_path,prediction,delimiter=",")
    print(f"Pridictions for {user_name} using {prediction_model_name} complete...")
    print(f"Predictions have been saved with names:")
    print(prediction_full_path)
    print(prediction_path)
    

    printOutput(prediction)
    
    

    # random_values=random_values.transpose()
    # prediction=prediction.transpose()
    # output=np.array([random_values])
    # print(random_values)
    # print(prediction)
    
    
def scale_values(array):
    # Find the highest and lowest values in the array
    highest = np.max(array)
    lowest = np.min(array)
    
    # Set the highest value to 5 and the lowest value to 0.5
    array[array == highest] = 5
    array[array == lowest] = 0.5
    
    # Scale values in between linearly between 0.5 and 5
    array = (array - lowest) / (highest - lowest) * 4.5 + 0.5
    
    return array
    

def writeResults(results,outputFile):
        np.savetxt(outputFile,results,delimiter=',',newline='')
        print(f"done writing to {outputFile}")
            
def get_npy_path():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Open file dialog and allow the user to select a file
    file_types = [("Npy Files", "*.npy")]
    file_path = filedialog.askopenfilename(filetypes=file_types)
    
    return file_path

def get_csv_path():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Open file dialog and allow the user to select a file
    file_types = [("csv Files", "*.csv")]
    file_path = filedialog.askopenfilename(filetypes=file_types)
    
    return file_path

def get_model_path():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Open file dialog and allow the user to select a file
    file_types = [("model Files", "*.keras")]
    file_path = filedialog.askopenfilename(filetypes=file_types)
    
    return file_path


def printOutput(data): # Select the top n recommendations, and format & print them
    n = 50
    selections = np.zeros((n, 2)) # This array holds the actual selected rating values, as well as their indexes within the output array
    print("please provide movies+links.csv")
    path=get_csv_path()
    with open(path, mode='r', encoding="utf-8") as file: # Match each rating to its respective movie
        reader = csv.reader(file)
        movies = list(reader)
        array = np.array(movies)

        for i, data2 in enumerate(data): # Loop over every rating in the NN output
            val = data2
            movieId = int(float(array[i+1][0]))
            selections = sorted(selections, key=lambda x: x[0]) # Sort the arrays by their rating values to make sure the least element of the array is at a predictable index
            if selections[0][0] < val: # If the current value is larger than the least element of the array, replace it
                selections[0][0] = val # Update the rating value
                selections[0][1] = movieId # Update the index of this rating

        print("\n=============== Neural Network Movie Recommendations ==============")

        count = 1
        for i in reversed(range(n)): # For each selected movie, from highest rating to least
            for x in range(1, len(data) - 1): # Loop over every rating in the NN output
                if int(array[x][0]) == int(selections[i][1]): 
                    line = array[x]
                    break
            #line = np.where(array[0] == int(selections[i][1]))
            print(f'Movie {count}: {line[1]}, Rating: {selections[i][0]}')
            count += 1

        
main()