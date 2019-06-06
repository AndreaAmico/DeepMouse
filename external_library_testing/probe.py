import tkinter as tk
from tkinter import ttk
import threading
import win32gui
import time
import numpy as np
from keras.models import load_model

class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.model = load_model('./model.pkl')
        self.prograss_bar = ttk.Progressbar()
        self.prograss_bar.pack(fill=tk.BOTH)

        left_guesses = 0
        for index in range(10):
            X_pred = self.get_batch()
            pred = self.model.predict_classes(X_pred)[0][0]
            left_guesses += pred
            print(f'\rRun_{index}: Current prediction = {"Left " if pred else "Right"}   '+
                f'Left probability = {left_guesses/index * 100:.1f}%   '+
                f'Right probability = {(1 - left_guesses/index) * 100:.1f}%', end='')

            # self.prograss_bar["value"]=(1 - left_guesses/index) * 100

        # self.measure_thread = threading.Thread(target=self.read_mouse_position, daemon=True)
        # self.measure_thread.start()

    def get_batch(self, batch_size=800): # 02
        X_pred = np.ones([1, batch_size, 2])
        for index in range(batch_size):
            x, y = win32gui.GetCursorPos()
            X_pred[0,index,:] = np.array([x, y])
            time.sleep(0.01)
            
        X_diff = X_pred[:,:-1,:].copy()
        X_diff[:,:,0] = np.diff(X_pred[:,:,0])
        X_diff[:,:,1] = np.diff(X_pred[:,:,1])
        
        x_std = 5. # np.mean(np.std(X_filt[:,:,0], axis=1))
        y_std = 6. # np.mean(np.std(X_filt[:,:,1], axis=1))

        X_diff[:,:,0] = X_diff[:,:,0] / x_std
        X_diff[:,:,1] = X_diff[:,:,1] / y_std
        return X_diff





if __name__ == "__main__":
    app = App()
    app.geometry("600x50")
    app.mainloop()
