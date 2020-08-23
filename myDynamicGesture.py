from myGlobalVariables import handGesDict

# 动态手势识别（手部姿势+运动）
# 输入手部姿势、手部运动、手部姿势置信度、手部运动置信度
# 输出手势识别结果（moveleft,moveright,movepp,movedown,turnleft,turnright,zoomin, zoomout,ok,cancel,close,invalid）及置信度
def gestureRecog(handGesture, movement, ch=1, cm=1):
    global handGesDict
    gesture = "invalid"
    # gesture = ""
    # gesture =movement
    confidence = ch * cm
    for h in handGesDict:
        if handGesture == h[0] and movement == h[1]:
            gesture = h[2]
            break
    return gesture, confidence