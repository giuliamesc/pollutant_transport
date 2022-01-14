u = [0.0598; 0.0575; 0.0543];
uex = 0.0615; % numerical solution
err = abs(u-uex);
dt = [1.25e-3; 5e-3; 1e-2];

figure
loglog(dt,err,'*-')
hold on
grid on
loglog(dt,0.5*dt)
loglog(dt,0.08*dt.^(1/2))
legend('error','sqrt(dt)','dt')