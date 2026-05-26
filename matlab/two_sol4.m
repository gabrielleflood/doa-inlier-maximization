function sol = two_sol4(r1,t1,r2,t2,ep)
    [sol1a,sol2a] =  two_sol(r1,t1,r2,t2,ep,ep);
    [sol1b,sol2b] =  two_sol(r1,t1,r2,t2,-ep,ep);
    [sol1c,sol2c] =  two_sol(r1,t1,r2,t2,ep,-ep);
    [sol1d,sol2d] =  two_sol(r1,t1,r2,t2,-ep,-ep);
    sol = [sol1a; sol2a; sol1b; sol2b;sol1c; sol2c; sol1d; sol2d];

    
    
