import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.datasets import make_classification
from keras.callbacks import EarlyStopping
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import sys
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("movies_metadata.csv")

#Summary our dataset
df.info()
#Shape of dataset
df.shape
#check the content of each column (first few rows)
df.head()

# delete the text columns that we dont need for the prediction
df.drop(['id','belongs_to_collection', 'homepage', 'imdb_id', 'original_language', 'original_title', 'overview', 'poster_path', 'production_companies', 'production_countries', 'runtime', 'release_date', 'spoken_languages', 'status', 'tagline', 'title', 'video','adult'], axis=1, inplace=True)
df.info()

#new shape after the delete
df[df['revenue'] == 0].shape
df['revenue'] = df['revenue'].replace(0, np.nan)
#replace all the non-numeric values with NaN
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['budget'] = df['budget'].replace(0, np.nan)
df[df['budget'].isnull()].shape

#A return value > 1 would indicate profit whereas a return value < 1 would indicate a loss.
df['return'] = df['revenue'] / df['budget']
df[df['return'].isnull()].shape

def clean_numeric(x):
    try:
        return float(x)
    except:
        return np.nan

df['popularity'] = df['popularity'].apply(clean_numeric).astype('float')
df["vote_count"] = df["vote_count"].apply(clean_numeric).astype('float')
df['vote_average'] = df['vote_average'].apply(clean_numeric).astype('float')


#check the number of missing values in the dataset
df.isnull().sum()
#Dropping rows with missing values
df.dropna(inplace=True)
#Check again
df.isnull().sum()
df["popularity"] = np.round(pd.to_numeric(df.popularity, errors='coerce'),2)
df.info()
df.shape

#converting 'genre' column into panda series and extract the type of the genre only from the column
s = pd.Series(df['genres'], dtype= str)
s1=s.str.split(pat="'",expand=True)
df['genre_ed']=s1[5]
#count of each genre in the dataset
df['genre_ed'].value_counts()
#Remove rows for genres with count less than 100
df=df[~df['genre_ed'].isin(['Mystery', 'Family', 'Documentary', 'War', 'Music', 'Western', 'History', 'Foreign', 'TV Movie'])]
#Drop original column from dataset
df.drop(['genres'], axis=1, inplace=True)
#get dummy columns for genre
df= pd.get_dummies(df, columns=["genre_ed"])
df.info()
df.head()

#Summary statistics of dataset
df.describe()

X,y = make_classification(n_features = 16, n_samples=312)
# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.20, shuffle = True, random_state = 42)
X_train
# Setting up the model
model = Sequential()
model.add(Dense(50,activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.4))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='Adam', loss='binary_crossentropy',
              metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10,
                   verbose=1,
                   restore_best_weights=True)

# Scalind the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size = 44,
                    verbose=1,
                    callbacks=[es])

history_dict = history.history
history_dict.keys()
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy of the model on test data: {:.2f}%".format(accuracy * 100))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

preds = np.round(model.predict(X_test),0)
confusion_matrix(y_test, preds)
print(classification_report(y_test, preds))

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageTk

# Function to handle the prediction based on user inputs
def predict_movie_success():
    try:
        budget = float(budget_entry.get())
        revenue = float(revenue_entry.get())
        popularity = float(popularity_entry.get())
        vote_average = float(vote_average_entry.get())
        vote_count = float(vote_count_entry.get())

        if budget <= 0 or revenue <= 0:
            messagebox.showerror("Error", "Please enter positive values for budget and revenue.")
            return

        return_on_investment = revenue / budget

        genre_input = genre_combobox.get()
        genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Fantasy', 'Horror', 'Romance', 'Science Fiction', 'Thriller']
        genre_dict = {genre: i for i, genre in enumerate(genres)}
        genre_array = np.zeros(len(genres))
        genre_array[genre_dict.get(genre_input, 0)] = 1

        features = np.array([[budget, popularity, vote_average, vote_count, return_on_investment] + list(genre_array)])
        features = scaler.transform(features)
        prediction = model.predict(features)
        predicted_class = np.round(prediction[0], 0)
        
        if return_on_investment > 1 or vote_average > 5:
            result_text = "The model predicts this movie will be a success."
        else:
            result_text = "The model predicts this movie will not be successful."

        messagebox.showinfo("Prediction Result", result_text)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

def on_closing():
    root.destroy()
    sys.exit()

class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.text = None

    def showtip(self, text):
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 27
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1, font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def createToolTip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

def main():
    global root, budget_entry, revenue_entry, popularity_entry, vote_average_entry, vote_count_entry, genre_combobox
    root = tk.Tk()
    root.title("Movie Success Prediction")
    root.geometry("500x400")

    # Load background image using Pillow
    image = Image.open('oscar.jpg.webp')  # Replace 'path_to_your_image.png' with your image file
    image = image.resize((500, 400), Image.LANCZOS)  # Resize the image to fit your window size
    background_image = ImageTk.PhotoImage(image)
    background_label = tk.Label(root, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Setup styles and other GUI elements
    style = ttk.Style()
    style.theme_use('default')
    
    # Customizing styles
    style.configure('TLabel', background='#f0f0f0', foreground='black', font=('Helvetica', 12, 'bold'))
    style.configure('TButton', font=('Helvetica', 12, 'bold'), borderwidth='1', relief='raised')
    style.map('TButton', background=[('active', '#0059b3'), ('disabled', '#d3d3d3')], foreground=[('active', 'white')])
    style.configure('TEntry', font=('Helvetica', 12), borderwidth='1')
    style.configure('TCombobox', font=('Helvetica', 12), borderwidth='1')

    # Custom frame for better appearance
    frame = tk.Frame(root, bg='#f0f0f0', padx=20, pady=20)
    frame.place(relx=0.5, rely=0.5, anchor='center')

    # Labels, entries, and button setup
    labels = ["Budget (in USD):", "Revenue (in USD):", "Popularity score:", "Average Vote (1-10 scale):", "Number of Votes:", "Genre:"]
    genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Fantasy', 'Horror', 'Romance', 'Science Fiction', 'Thriller']
    entries = []
    for i, label_text in enumerate(labels):
        label = ttk.Label(root, text=label_text)
        label.grid(row=i, column=0, padx=10, pady=10, sticky='e')
        if label_text == "Genre:":
            entry = ttk.Combobox(root, values=genres, state='readonly')
            entry.set('Select Genre')
        else:
            entry = ttk.Entry(root)
        entry.grid(row=i, column=1, padx=10, pady=10, sticky='ew')
        createToolTip(entry, f"Enter the {label_text.lower()}")
        entries.append(entry)

    budget_entry, revenue_entry, popularity_entry, vote_average_entry, vote_count_entry, genre_combobox = entries

    predict_button = ttk.Button(root, text="Predict", command=predict_movie_success)
    predict_button.grid(row=6, column=0, columnspan=2, padx=10, pady=20)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
