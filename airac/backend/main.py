from retrieval_pipeline import Badal
from retrieval_pipeline import ChatState

badal = Badal()

while True:
    user = input("[USER] ")
    if (user.lower() == "quit"):
        break
    answer = badal.invoke(user)
    print(f"[BADAL] {answer}")