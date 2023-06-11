
%% R tipping in a turbulent thermoacoustic system
% This code is written by Induja
% Date: 20/02/2021 (last updated 10.06.2023)
% -----------------------------------------------------

%% 
% plotting time series
clear
figure;
pm = load('1.txt');
plot(pm(:,1)-pm(1,1),pm(:,3).* (1000/0.2175))
%%
% input 
clearvars -except R
ri = 5;
p1= load('1.txt');
data_total = p1;
m_f = 24; % mass flow rate of fuel slpm
flow1 = 380; flow2 = 640; % mass flow rates of air
R.Time(ri,1) =140; % duration in s

R.Rate(ri,1) = (flow2-flow1)/R.Time(ri,1); % rate
rate = R.Rate(ri,1); T =R.Time(ri,1);
column_no = 3; % select pressure transducer
fs =4000;   % sampling freq
conv_f = (1000/0.2175);% convertion factor to pascal
w = fs/1;     % window size
w_skip = fs/10;    % no. of points to skip
f = 190;       % freq in Hz
% n1 = 2;       % no. cycles
% n2 = 4;

data = data_total(:,column_no).*conv_f; % convert to pascal
time =  data_total(:,1)-data_total(1,1);
data = data- mean(data);
L=length(data); 
% calculate eq ratio
t = [1/fs:1/fs:T];
ma = flow1+rate.*t ;
eq_r = (m_f./(ma.*1)).*15.8;

% Reynolds number
m_a = ma;
slpm = m_a + m_f; % total flow rate
rho = 1.225; mu = 2*10^-5; 
D = 40*10^-3; d0 = 16*10^-3;
m_dot = (rho.*slpm./60000); % mass flow rate
% Re = (4*m_dot)./(pi*mu*(D+d0)); %Bluff body
Re = (4*m_dot)./(pi*mu*(D)); % Re for swirl

% maxlag = 2*floor(fs/f);
subplot(2,1,1) % plotting
p1 = plot(time,data);
set(gca,'FontSize',12,'LineWidth',1,'FontWeight','normal');
xlabel('t (s)','FontSize',14);
ylabel('p^\prime','FontSize',14); 

   k=[1:w_skip:L-w+1];
    n=length(k);
    
parfor k1 =1:n-1
    datawin(k1,:)= k(k1):k(k1)+w-1;
    wdata=data(datawin(k1,:));    
    p=wdata(:);
    p=p(:);
    prms(k1) = rms(p);
    eq_ratio(k1) = eq_r(k(k1)+w-1);
    Re_1(k1) = Re(k(k1)+w-1);
end

time_wind = [w:w_skip:L-1]./fs;
% wp = 2000/1; w_skip_p = wp/10; 
% Lp = length(m_a); 
% j=1:w_skip_p:Lp-wp;
%  for i=1:length(j)
%      ma(i)=mean(m_a(j(i):j(i)+wp-1,2));
%  end


subplot(2,1,2)
% plot(Re_1,prms)
p2 = plot(time_wind ,prms);
% set(gca,'Xdir','reverse')
set(gca,'FontSize',12,'LineWidth',1,'FontWeight','normal');
xlabel('Re','FontSize',14);
ylabel('p^\prime','FontSize',14); 
R.PRMS {1,ri,:} = prms;
R.Rate (ri,1) = rate;
R.Time (ri,1) = T;
% R.eq_r {1,ri,:}  = eq_r;
% R.eq_ratio {1,ri,:}  =eq_ratio;
R.Re {1,ri,:}  = Re;
R.Re1 {1,ri,:}  = Re_1;
R.W = w;
R.w_skip = w_skip;
R.rate_Re = (Re(end)-Re(1))./R.Time(:,1);

R.T_ind{1,ri,:} = diff(R.PRMS {1,ri,:});
% p = plot(fseg(1,:),pseg(:,2),'r', 'LineWidth',5);
%// modified jet-colormap
n = length(prms);
cd = [uint8(jet(n)*255) uint8(ones(n,1))].'; %'
drawnow
set(p2.Edge, 'ColorBinding','interpolated', 'ColorData',cd)

%% Read temperature

clearvars -except R
ri =10;
p1= load('1.txt');
data_total = p1;
m_f = 24; %slpm
flow1 = 380; flow2 = 640;
rate = R.Rate(ri,1); T = R.Time(ri,1);
column_no1 = 2; column_no2 = 3; 

fs =4;   % sampling freq
w = fs/1;     % window size
w_skip = fs/4;    % no. of points to skip
f = 190;       % freq in Hz
t = [1/fs:1/fs:T];
% calculate eq ratio
ma = flow1+rate.*t ;

% ramp_stop = 549%length(t); %time step
% data1 = data_total(ramp_stop-length(t)+1:ramp_stop,column_no1); 
% data2 = data_total(ramp_stop-length(t)+1:ramp_stop,column_no2); 
data1 = data_total(:,column_no1); 
data2 = data_total(:,column_no2);
L=length(data1); 

% Reynolds number
m_a = ma;
slpm = m_a + m_f; % total flow rate
rho = 1.225; mu = 2*10^-5; 
D = 40*10^-3; d0 = 16*10^-3;
m_dot = (rho.*slpm./60000);
% Re = (4*m_dot)./(pi*mu*(D+d0)); %Bluff body
Re = (4*m_dot)./(pi*mu*(D)); % swirl

% R.temp {1,ri,:}  = data1;
% R.temp {2,ri,:}  = data2;
% R.Re_T {1,ri,:}  = Re;

%% plot temp
figure;
%redblu colorbar
cc= [[103,0,31];[178,24,43];[214,96,77];[244,165,130];[253,219,199];[209,229,240];[146,197,222];[67,147,195];[33,102,172];[5,48,97]]./255;
 
for ri = 1:10
y1= R.temp{1,ri,:};
y2= R.temp{2,ri,:};
x = R.Re_T{1,ri,:};
c1 = [1:-0.09:0];
c2 = [0:0.09:1];

% plot(x(1:length(y1)),y1,'o','MarkerSize',3,'MarkerEdgeColor',[0 c2(ri) 0.8 ],'MarkerFaceColor',[0 c2(ri) 0.8]);
plot(x(1:length(y1)),y2,'o','MarkerSize',3,'MarkerEdgeColor',cc(ri,:),'MarkerFaceColor',cc(ri,:));

hold on
end

%% Reynolds number
m_a = ma;
m_f = 24; % fuel flow rate in slpm
slpm = m_a + m_f; % total flow rate
rho = 1.225; mu = 2*10^-5; 
D = 40*10^-3; d0 = 16*10^-3;
m_dot = (rho.*slpm./60000);
% Re = (4*m_dot)./(pi*mu*(D+d0)); %Bluff body
Re = (4*m_dot)./(pi*mu*(D)); % swirl

%% waterfall prms, colour with time

frange=[1000,2000]; % multiples of 4
fs = 4000;
j= R.Time;%[0:0.1:250];
figure;
ReMax = R.Re1{1,1};
% f_res = 0.1;
for fileNumb1 = 1:11%length(names)
    f = R.Re1{1,fileNumb1};
    L = length(f);
    %L = length(ReMax );
     i=fileNumb1;
    jj=j(fileNumb1);
    X = ones(1,L)*jj;
    P2 = R.PRMS{1,fileNumb1};
    P3 = interp1(f,P2,ReMax);
%     PX (fileNumb1 ,:) = P3;
    X= X(:);
    f3 = ReMax(:);
    P3=P3(:);
%     xseg = [X(1:end-1),X(2:end)];
%     fseg = [f(1:end-1),f(2:end)];
%     pseg = [P3(1:end-1),P3(2:end)];
%     % Plot all line segments (invisible for now unless you remove 'visible','off')
%     h = plot3(fseg',xseg',pseg','-','LineWidth',4);
%     %     xlim([min(x) max(x)]);
%     %     ylim([min(y) max(y)]);
%     segColors = jet(size(pseg,1)); % Choose a colormap
%     set(h, {'Color'}, mat2cell(segColors,ones(size(pseg,1),1),3));%mat2cell(segColors,ones(size(pseg,1),1),3))
    c1 = [1:-0.09:0];
    c2 = [0:0.09:1];
 
    plot3(f,X,P2,'o','MarkerSize',3,'MarkerEdgeColor',[0 c2(fileNumb1) 0.8],'MarkerFaceColor',[0 c2(fileNumb1) 0.8]);

    hold on;
    %set(gca,'FontSize',12,'LineWidth',1,'FontName','Trebuchet MS','FontWeight','normal','XLim',[0 1000],'YLim',[0 0.25]);
    %      set(gca,'FontSize',12,'LineWidth',1,'XLim',[0 500]);[c2(fileNumb1) c1(fileNumb1) 0.6],'MarkerFaceColor',[c2(fileNumb1) c1(fileNumb1) 0.6]
     xlabel('$Re$','FontSize',12);ylabel('Total time','FontSize',12);
     zlabel('$p\prime_{rms}$','FontSize',12);

end
%     xlin = linspace(min(R.Rate),max(R.Rate),20);
%     ylin = linspace(min(R.Re1{1,11}),max(y),3240);
%% decay rate vs Temp

% D =mean_D'; er = er_D;
D =D2'; er = er2; T = T2;

errorbar(T(1:end,1),1.*D(1:end),er(1:end),'.r')
plot((T(1:end,1)),(1.*D(1:end)),'.r')

% fitting semilog
rw = T(1:end,1); 
s = -1.*D(1:end);
figure;
fitobject = fit((rw),log(s'),'poly1');
plot(fitobject,(rw),log(s))
text(min((rw(:))),max(log(s(:))),sprintf('Slope=%f',fitobject.p1))
ci = confint(fitobject,0.95);
mean_s=(ci(1,1)+ci(2,1))/2
err=ci(2,1)-(ci(1,1)+ci(2,1))/2;
c = fitobject.p2;
% 
% figure; 
AN = s;%max(FFT_peak);
liney=mean_s.*(rw(:))+c;
plot(rw,-1.*(exp(liney)),'k','LineWidth',1.5)
hold on
plot(rw,-1.*AN,'.b','MarkerSize',23)
set(gca,'FontSize',12,'LineWidth',1,...
           'FontName','Trebuchet MS','FontWeight','normal');
 set(gca,'YScale', 'log','FontSize',14,'LineWidth',1);%,...
%            'XLim',[0 100],'YLim',[0 10],'FontName','Trebuchet MS','FontWeight','normal');       
xlabel('\mu_2\mu_0','FontSize',14);% Characteristic path length TransitivityNode strength
ylabel('P','FontSize',14); 
text(min((rw(:))),-1*max((s(:))),sprintf('Slope=%f+%f',fitobject.p1,err))

fitobject = fit((rw),(s'),'poly1');
plot(fitobject,(rw),(s))
ci = confint(fitobject,0.95);
mean_s=(ci(1,1)+ci(2,1))/2
err=ci(2,1)-(ci(1,1)+ci(2,1))/2;
text(min((rw(:))),max((s(:))),sprintf('Slope=%f+%f',fitobject.p1,err))
c = fitobject.p2;
%% surface plot
%  
%  [X,Y] = meshgrid(R.Rate,ReMax);
%  s = surf(X,Y,PX','FaceAlpha',0.8);
% s.EdgeColor = 'none';    

%% pdf of data

clearvars -except R
ri =1;
p1= load('1.txt');
data_total = p1;
m_f = 24; %slpm
flow1 = 380; flow2 = 640;
R.Rate(ri,1) = 1.625;R.Time(ri,1) = 160;
rate = R.Rate(ri,1); T =R.Time(ri,1);
column_no = 3; 
fs =4000;   % sampling freq
conv_f = (1000/0.2175);% convert to pascal
w = fs/1;     % window size
w_skip = fs/10;    % no. of points to skip
f = 190;       % freq in Hz
% n1 = 2;       % no. cycles
% n2 = 4;

data = data_total(:,column_no).*conv_f; % convert to pascal
time =  data_total(:,1)-data_total(1,1);
data = data- mean(data);
L=length(data); 
% calculate eq ratio
t = [1/fs:1/fs:T];
ma = flow1+rate.*t ;
eq_r = (m_f./(ma.*1)).*15.8;

% Reynolds number
m_a = ma;
slpm = m_a + m_f; % total flow rate
rho = 1.225; mu = 2*10^-5; 
D = 40*10^-3; d0 = 16*10^-3;
m_dot = (rho.*slpm./60000);
% Re = (4*m_dot)./(pi*mu*(D+d0)); %Bluff body
Re = (4*m_dot)./(pi*mu*(D)); % swirl

% maxlag = 2*floor(fs/f);
subplot(2,1,1)
plot(time,data)
set(gca,'FontSize',12,'LineWidth',1,'FontWeight','normal');
xlabel('t (s)','FontSize',14);% Characteristic path length TransitivityNode strength
ylabel('p^\prime','FontSize',14); 

   k=[1:w_skip:L-w+1];
    n=length(k);
    hurs=zeros(1,n-1);
parfor k1 =1:n-1
    datawin(k1,:)= k(k1):k(k1)+w-1;
    wdata=data(datawin(k1,:));    
    p=wdata(:);
    p=p(:);
    prms(k1) = rms(p);
    eq_ratio(k1) = eq_r(k(k1)+w-1);
    Re_1(k1) = Re(k(k1)+w-1);
end

%
%% waterfall prms, colour with time

frange=[1000,2000]; % multiples of 4
fs = 4000;
j= R.Time;%[0:0.1:250];
figure;
ReMax = R.Re1{1,1};
% f_res = 0.1;
for fileNumb1 = 1%length(names)
    f = R.Re1{1,fileNumb1};
    L = length(f);
    %L = length(ReMax );
     i=fileNumb1;
    jj=j(fileNumb1);
    X = ones(1,L)*jj;
    P2 = R.PRMS{1,fileNumb1};
    P3 = interp1(f,P2,ReMax);
%     PX (fileNumb1 ,:) = P3;
    X= X(:);
    f3 = ReMax(:);
    P3=P3(:);
    xseg = [X(1:end-1),X(2:end)];
    fseg = [f(1:end-1),f(2:end)];
    pseg = [P3(1:end-1),P3(2:end)];
    % Plot all line segments (invisible for now unless you remove 'visible','off')
    h = plot3(fseg',xseg',pseg','-','LineWidth',4);
    h = plot(fseg(1,:),pseg(:,2),'-','LineWidth',4);
    %     xlim([min(x) max(x)]);
    %     ylim([min(y) max(y)]);
    segColors = jet(size(pseg,1)); % Choose a colormap
    set(h, {'Color'}, mat2cell(segColors,ones(size(pseg,1),1),3));%mat2cell(segColors,ones(size(pseg,1),1),3))
    c1 = [1:-0.09:0];
    c2 = [0:0.09:1];
 
    plot3(f,X,P2,'o','MarkerSize',3,'MarkerEdgeColor',[0 c2(fileNumb1) 0.8],'MarkerFaceColor',[0 c2(fileNumb1) 0.8]);

    hold on;
    %set(gca,'FontSize',12,'LineWidth',1,'FontName','Trebuchet MS','FontWeight','normal','XLim',[0 1000],'YLim',[0 0.25]);
    %      set(gca,'FontSize',12,'LineWidth',1,'XLim',[0 500]);[c2(fileNumb1) c1(fileNumb1) 0.6],'MarkerFaceColor',[c2(fileNumb1) c1(fileNumb1) 0.6]
     xlabel('$Re$','FontSize',12);ylabel('Total time','FontSize',12);
     zlabel('$p\prime_{rms}$','FontSize',12);

end
%     xlin = linspace(min(R.Rate),max(R.Rate),20);
%     ylin = linspace(min(R.Re1{1,11}),max(y),3240);
%% coloured figure
figure;
n = 589;
% x = linspace(-10,10,n); y = x.^2;
p = plot(fseg(1,:),pseg(:,2),'r', 'LineWidth',5);
%// modified jet-colormap
cd = [uint8(copper(n)*255) uint8(ones(n,1))].'; %'
drawnow
set(p.Edge, 'ColorBinding','interpolated', 'ColorData',cd)
% ---------------------------------------------------------------- 
%% Time axis on top

t=time;
tt = time_wind;
%
set(gca,'box','off')
isholdonque = ishold;
hold on
ax = axis;
plot(ax(2)*[1,1],ax(3:4),'k','linewidth',1.5)
plot(ax(1:2),ax(4)*[1,1],'k','linewidth',1.5)
if isholdonque == 0
    hold off
end
%
ax1=gca;
ax2 = axes('position', get(ax1, 'position'));
%to add a new x-axis on top of previous one
set(ax2, 'XAxisLocation', 'Top', 'color', 'none', 'Xdir','normal');
ax2.YAxis.Visible = 'off';
%

Re_plus = (Re(end)-Re(1))/4; 

set(gca,'XTick',[0:0.25:1]); set(gca,'TickDir','out');ax = gca; ax.TickLength = [0.02 0.02];ax.LineWidth = 1.5;
set(gca,'XTickLabel',[ ] );
format short
set(gca,'XTickLabel',ceil([Re(1):Re_plus:Re(end)]));

%% rate of Re axis

set(gca,'box','off')
isholdonque = ishold;
hold on
ax = axis;
plot(ax(2)*[1,1],ax(3:4),'k','linewidth',1.5)
plot(ax(1:2),ax(4)*[1,1],'k','linewidth',1.5)
if isholdonque == 0
    hold off
end
ax1=gca;
ax2 = axes('position', get(ax1, 'position'));
%to add a new x-axis on top of previous one
set(ax2, 'YAxisLocation', 'Right', 'color', 'none', 'Ydir','normal');
ax2.XAxis.Visible = 'off';

set(gca,'YTick',[0.1:0.1:1]); set(gca,'TickDir','out');ax = gca; ax.TickLength = [0.02 0.02];ax.LineWidth = 1.5;
set(gca,'YTickLabel',[ ] );
format short
set(gca,'YTickLabel',ceil([R.rate_Re(1:end)]));

%% plot timing index
for ri = 1:4
x=cell2mat(R.Re1(1,ri,:));
y = R.T_ind{1,ri,:}+ri*0*200;
plot(x(2:end),y(1:end))
hold on
end
%%

j= R.Time;%[0:0.1:250];
figure;
for fileNumb1 = 1:10%length(names)
    f = R.Re1{1,fileNumb1};
    L = length(f);
     i=fileNumb1;
    jj=j(fileNumb1);
    X = ones(1,L)*jj;
    P2 = R.T_ind{1,fileNumb1};
    X= X(:);
    c1 = [1:-0.09:0];
    c2 = [0:0.09:1];
 
    plot3(f(2:end),X(2:end),P2,'o','MarkerSize',3,'MarkerEdgeColor',[0 c2(fileNumb1) 0.8],'MarkerFaceColor',[0 c2(fileNumb1) 0.8]);

    hold on;
    %set(gca,'FontSize',12,'LineWidth',1,'FontName','Trebuchet MS','FontWeight','normal','XLim',[0 1000],'YLim',[0 0.25]);
    %      set(gca,'FontSize',12,'LineWidth',1,'XLim',[0 500]);[c2(fileNumb1) c1(fileNumb1) 0.6],'MarkerFaceColor',[c2(fileNumb1) c1(fileNumb1) 0.6]
     xlabel('$Re$','FontSize',12);ylabel('Total time','FontSize',12);
     zlabel('$p\prime_{rms}$','FontSize',12);

end


% #383444,#3A4757,#395B67,#376F72,#3D8378,#4E9778,#6AA974,#8EBA6D,#B8C869,#E6D469
% #333442,#364655,#355A64,#356E6F,#3D8274,#4F9675,#6BA871,#8FB96C,#B8C868,#E6D469
% #AD595F,#87C4F3 #293E4F,#B9605D D16E6E