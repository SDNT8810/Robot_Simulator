%% Initialize the Parameters
clear;clc
NL = 360; % Number of Lidar beams
Xmax = 2; % Lidar beams sight length
Lidar_Degrees(:,1) = (1:NL)-180; % The angles of Lidar beams in Degrees
Lidar_Radians(:,1) = deg2rad(Lidar_Degrees); % The angles of Lidar beams in Degrees
Lidar_Beams = ones(NL,1)*Xmax; % Lidar Beams

Ns = 12; % Number of the Sections
Sec_Deg = linspace(0,NL,Ns+1)-180; % Degrees of the center of sections
Sec_Deg(1) = [];
%% Lidar and Obstacles
Obs_Number = 3; % Number of the Obstacles
Obs_Width = [20 10 30]; % The width of the obstacles
Obs_Distance = [0.5 0.8 0.2]; % Distance of the obstacles to the robot (Normalize between 0 and 1)
Obs_Deg = [0 80 190]; % Degree in which the obstacles are
for i = 1:Obs_Number
  if Obs_Deg(i)==0; Obs_Deg(i) = 360; end
  Obs_Left = rem(Obs_Deg(i)+180-Obs_Width(i), 360); % Number of the beam shows the Left side of the Obstacle
  Obs_right = rem(Obs_Deg(i)+180+Obs_Width(i), 360); % Number of the beam shows the Right side of the Obstacle
  if Obs_Left<Obs_right
    Lidar_Beams(Obs_Left:Obs_right) = Obs_Distance(i)*Xmax;
  else
    Lidar_Beams(Obs_Left:end) = Obs_Distance(i)*Xmax;
    Lidar_Beams(1:Obs_right) = Obs_Distance(i)*Xmax;
  end
end
%% Sections and Gaussian Membership Functions
std = 20; % Standard Deviation of the Gaussian MF
wij = @(deg,Center)(gaussmf(deg,[std,Center]));
W2 = zeros(Ns,NL); % Memberships of all the beams in all the sections
Mu = []; % The value of the Obs in each Section
Sec_Free_Val = zeros(1,Ns); % The value of each Section
for i = 1:Ns
  Center = Sec_Deg(i);
  W2(i,:) = wij(wrapTo180(Lidar_Degrees-Center),0);
  Mu = W2(i,:)*Lidar_Beams/Xmax; 
  Sec_Free_Val(i) = Mu/sum(W2(i,:));
end
%% Goal Direction Consideration Function
G_dirc = -160;
G_STD = 70;
wgj = @(j)(gaussmf(Sec_Deg(j),[G_STD,G_dirc]));
Goal_Dir_Value = zeros(1,Ns); % The Value of each section considering the Obs and Goal
for j = 1:Ns
  Goal_Dir_Value(j) = wgj(j)*Sec_Free_Val(j);
end
%% Goal Distance Consideration Function
G_dist = 2;
mu_close = zmf(G_dist, [0, 1]);
mu_near = gaussmf(G_dist, [1, 2]);
mu_far = smf(G_dist, [2, 5]);
M = [mu_close; mu_near; mu_far]

    %% Replot and update GUIs
    figure(2)
    clf
    
    xlim([-180, 180])
    
    subplot(221); 
    polarplot(deg2rad([-180 Sec_Deg]), [Sec_Free_Val(end), Sec_Free_Val],'LineWidth',2)
    hold on
    polarplot(Lidar_Radians, Lidar_Beams/Xmax,'--','LineWidth',2)
    polarplot(deg2rad([-180 Sec_Deg]), [Goal_Dir_Value(end), Goal_Dir_Value],'k','LineWidth',2);
    polarplot(deg2rad([0 G_dirc]), [0 G_dist]/Xmax,'R','LineWidth',2);
    polarfill(gca,Lidar_Radians,Lidar_Beams*0,Lidar_Beams/Xmax,'blue',0.15)
    % title('Obs. and MFs')
    
    subplot(222); hold on
    plot([G_dirc G_dirc],[0 1],'Color','r','LineWidth',2,'Marker','*');
    plot(Lidar_Degrees, Lidar_Beams/Xmax,'r--','LineWidth',2)
    area(Lidar_Degrees, Lidar_Beams/Xmax,'FaceColor','b','FaceAlpha',.15,'EdgeAlpha',.15)
    plot([-180 Sec_Deg], [Sec_Free_Val(end), Sec_Free_Val],'color','b','LineWidth',2)
    plot([-180 Sec_Deg], [Goal_Dir_Value(end), Goal_Dir_Value],'k','LineWidth',2);
    area([-180 Sec_Deg], [Goal_Dir_Value(end), Goal_Dir_Value],'FaceColor','k','FaceAlpha',.15,'EdgeAlpha',.15)
    xlim([-180, 180]); box on
    
    subplot(212)
    hold on
    for i = 1:NL
        line([Lidar_Degrees(i) Lidar_Degrees(i)], [0 Lidar_Beams(i)],'color',[0.8,0.8,.9])
    end
    O(1) = plot([G_dirc G_dirc],[0 Xmax],'Color','r','LineWidth',2,'Marker','*');
    O(2) = scatter([-180 Sec_Deg], [Sec_Free_Val(end), Sec_Free_Val],15,[0.9,0.5,0.5],'filled','DisplayName','Fuz_Lidar');
    O(3) = plot([-180 Sec_Deg], [Sec_Free_Val(end), Sec_Free_Val],'r');
    O(4) = scatter([-180 Sec_Deg], [Goal_Dir_Value(end), Goal_Dir_Value],15,'b','filled','DisplayName','Goal Direction Considered');
    O(5) = plot([-180 Sec_Deg], [Goal_Dir_Value(end), Goal_Dir_Value],'b');
    box on; xlim([-180 180])
    % title('Informations')

%% Required Functions
function polarfill(ax_polar,theta,rlow,rhigh,color,alpha)
    ax_cart = axes();
    ax_cart.Position = ax_polar.Position;
    [xl,yl] = pol2cart(theta,rlow);
    [xh,yh] = pol2cart(fliplr(theta),fliplr(rhigh));
    fill([xl,xh],[yl,yh],color,'FaceAlpha',alpha,'EdgeAlpha',0);
    xlim(ax_cart,[-max(get(ax_polar,'RLim')),max(get(ax_polar,'RLim'))]); 
    ylim(ax_cart,[-max(get(ax_polar,'RLim')),max(get(ax_polar,'RLim'))]);
    axis square; set(ax_cart,'visible','off');
end

