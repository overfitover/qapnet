Var1 = 1
Var2 = 0
def function():
    if Var2 == 0 and Var1 > 0:
        print("Result One")
    elif Var2 == 1 and Var1 > 0:
        print("Result Two")
    elif Var1 < 1:
        print("Result Three")
    Var1 =- 1
function()