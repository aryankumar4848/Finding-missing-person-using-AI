try:
    import mediapipe as mp
    try:
        from mediapipe.python.solutions import face_mesh as fm
        print("left", fm.FACEMESH_LEFT_EYE)
    except:
        pass
except BaseException as e:
    print(e)
