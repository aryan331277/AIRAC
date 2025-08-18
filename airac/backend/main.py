from backend.retrieval_pipeline import Badal, ChatState


badal = Badal()

while True:
    user = input("[USER] ")
    if (user.lower() == "quit"):
        break
    answer = badal.invoke(user)
    print(f"[BADAL] {answer}")