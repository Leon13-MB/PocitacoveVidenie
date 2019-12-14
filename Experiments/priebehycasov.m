clear;clc;
dt_cas_vf = readtable('D:\Python\VlastnaFunkciaTest.xlsx');
dt_cas_ocvf = readtable('D:\Python\OpenCVtest2.xlsx');
s=size(dt_cas_vf);
t=1:1:s(1);
x=dt_cas_ocvf{:,1};
x2=dt_cas_vf{:,1};
plot(t,x,t,x2)
xlim([0 1020])
ylim([0 100])
grid on
ylabel('èas[ms]')
xlabel('poèet experimentov')
legend('cas opencv','cas vlastnaf')
title('Èas výpoètu algoritmu medzi framom 10 a frameom 11')
set(gca,'ytick',linspace(0,100,21))
saveas(gcf,'D:\\Python\CasyFunkcii.png')
%T-TEST, H0:F(X) = G(X), H1: F(X)!=G(X), alfa = 0.05
if ttest2(x2,x,'Tail','both','Vartype','equal')
    disp("Na danej hladine alfa," +...
        "sa casy oboch funkcii vyznamne lisia.");
else
    disp("Na danej hladine alfa," +...
        "sa casy oboch funkcii vyznamne nelisia.");
end


