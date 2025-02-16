import pyttsx3

engine = pyttsx3.init()
greet_people = set()

if confidence > 0.8 and name not in greet_people:
    greet = f"Hello {name}"
    print(greet)
    engine.say(greet)
    engine.runAndWait()
    greet_people.add(name)

    