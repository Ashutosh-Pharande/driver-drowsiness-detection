import threading
import winsound  # Windows only


alarm_playing = False


def play_alarm():
    """
    Plays a beep sound continuously in a separate thread.
    """
    global alarm_playing
    while alarm_playing:
        winsound.Beep(1000, 500)  # frequency=1000Hz, duration=500ms


def start_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        threading.Thread(target=play_alarm, daemon=True).start()


def stop_alarm():
    global alarm_playing
    alarm_playing = False