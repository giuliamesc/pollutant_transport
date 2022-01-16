u = [0.05947; 0.05784; 0.05558; 0.05274]; % stochastic solutions
uex = 0.0615; % numerical solution
err = abs(u-uex);
dt = [1.25e-3; 2.5e-3; 5e-3; 1e-2]; % vector of times

figure
loglog(dt,err,'*-')
hold on
grid on
loglog(dt,0.7*dt,'*-')
loglog(dt,0.1*dt.^(1/2),'*-')
legend('error','dt','sqrt(dt)')