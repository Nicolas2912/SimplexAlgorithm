import gurobipy as GRB
l = GRB.Model("Leisten")
leisten = {i: 6.5 for i in range(300)}
leisten.update({i: 5.5 for i in range(300, 380)})
# Alternative if GRB.INTEGER fails despite correct installation:
x = l.addVars([1.5, 2], leisten, vtype='I') 
# x = l.addVars([1.5, 2], leisten, vtype=GRB.INTEGER) # Keep the standard way first commented out for now
l.addConstrs(1.5 * x[1.5, i] + 2 * x[2, i] <= leisten[i] for i in leisten)
l.addConstr(sum(x[1.5, i] for i in leisten) == 2 * sum(x[2, i] for i in leisten))
# Use -1 for maximization as GRB.MAXIMIZE is not found
l.setObjective(sum(x[2, i] for i in leisten), -1) 
# l.setObjective(sum(x[2, i] for i in leisten), GRB.MAXIMIZE) # Original line commented out
l.optimize()
muster_lang = {}
muster_kurz = {}
for i in leisten:
    muster = muster_lang if leisten[i] == 6.5 else muster_kurz
    if (x[1.5, i].x, x[2, i].x) in muster:
        muster[(x[1.5, i].x, x[2, i].x)] += 1
    else:
        muster[(x[1.5, i].x, x[2, i].x)] = 1
print("6.5er Leisten:", muster_lang)
print("5.5er Leisten:", muster_kurz)
