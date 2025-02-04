import tkinter as tk
def center(w):
    width=w.winfo_width()
    height=w.winfo_height()
    sc_w=w.winfo_screenwidth()
    sc_h=w.winfo_screenheight()
    x=(sc_w-width)//2
    y=(sc_h-height)//2
    w.geometry(f'{width}x{height}+{x}+{y}')
def GUI():
    window = tk.Tk()
    window.wm_attributes("-topmost",True)
    window.wm_attributes("-alpha",0.9)
    center(window)
    window.geometry('800x100')
    first=tk.Label(text="Who do you want to call?")
    first.pack()
    window.after(5000, lambda: window.destroy())
    window.mainloop()

def main():
    GUI()
if __name__ == "__main__":
    main()
    

