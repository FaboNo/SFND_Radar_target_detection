clear all
clc

%% Radar Specifications 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77GHz
% Max Range = 200m
% Range Resolution = 1 m
% Max Velocity = 100 m/s
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%speed of light = 3e8
%% User Defined Range and Velocity of target
% *%TODO* :
% define the target's initial position and velocity. Note : Velocity
% remains contant
 
Range_target = 125;
V_target = 10;

%% FMCW Waveform Generation

% *%TODO* :
%Design the FMCW waveform by giving the specs of each of its parameters.
% Calculate the Bandwidth (B), Chirp Time (Tchirp) and Slope (slope) of the FMCW
% chirp using the requirements above.


%Operating carrier frequency of Radar 
fc= 77e9;             %carrier freq Hz

Range_max = 200;  % (m)
Range_res = 1;    % (m)
V_max = 70;       % (m/s)
V_res = 3;        % (m/s)
c = 3e8;
B_sweep = c / (2 * Range_res);      % compute Bsweep
T_chirp = 5.5 * 2 * Range_max / c;
slope = B_sweep / T_chirp;
                                                          
%The number of chirps in one sequence. Its ideal to have 2^ value for the ease of running the FFT
%for Doppler Estimation. 
Nd=128;                   % #of doppler cells OR #of sent periods % number of chirps

%The number of samples on each chirp. 
Nr=1024;                  %for length of time OR # of range cells

% Timestamp for running the displacement scenario for every sample on each
% chirp
% start 0 length Nd*Tchirp  Nr*Nd samples
t=linspace(0, Nd*T_chirp, Nr*Nd); 

%Creating the vectors for Tx, Rx and Mix based on the total samples input.
Tx = zeros(1,length(t)); %transmitted signal
Rx = zeros(1,length(t)); %received signal
Mix = zeros(1,length(t)); %beat signal

%Similar vectors for range_covered and time delay.
r_t = zeros(1,length(t)); % range covered by the target
td = zeros(1,length(t));


%% Signal generation and Moving Target simulation
% Running the radar scenario over the time. 

for i=1:length(t)         
    
    
    % *%TODO* :
    % For each time stamp update the Range of the Target for constant velocity. 
    % x(t) = x(t-1) + V*DeltaT
    
    r_t(i) = Range_target+ V_target*(t(i));
    
    % compute tau
    td(i) = 2*r_t(i)/c;
    
    % *%TODO* :
    %For each time sample we need update the transmitted and
    %received signal.  Apply equations provided in the Section 2
    Tx(i) = cos(2*pi*(fc*t(i)+0.5*slope*t(i)^2));
    Rx (i) = cos(2*pi*(fc*(t(i)-td(i))+0.5*slope*(t(i)-td(i))^2));
    
    % *%TODO* :
    %Now by mixing the Transmit and Receive generate the beat signal
    %This is done by element wise matrix multiplication of Transmit and
    %Receiver Signal
    Mix(i) = Tx(i) .* Rx(i);
    
end

%% RANGE MEASUREMENT

% *%TODO* :
%reshape the vector into Nr*Nd array. Nr and Nd here would also define the size of
%Range and Doppler FFT respectively.
% get a matrix with Nr row / Nd column
Mix = reshape(Mix, [Nr, Nd]);

% *%TODO* :
%run the FFT on the beat signal along the range bins dimension (Nr) and
%normalize.
% According to documentation to take fft on range is: (by default take the
% first colomn
mix_fft = fft(Mix);

% *%TODO* :
% Take the absolute value of the normalized FFT output
P2 = abs(mix_fft / max(mix_fft));

% *%TODO* :
% Output of FFT is double sided signal, but we are interested in only one side of the spectrum.
% Hence we throw out half of the samples.
P1 = P2(1 : Nr/2 + 1);

% frequency range
f = (0 : (Nr / 2));
plot(f, P1);
title("Range display");
axis ([0 200 0 1]);


%% RANGE DOPPLER RESPONSE
% The 2D FFT implementation is already provided here. This will run a 2DFFT
% on the mixed signal (beat signal) output and generate a range doppler
% map.You will implement CFAR on the generated RDM


% Range Doppler Map Generation.

% The output of the 2D FFT is an image that has reponse in the range and
% doppler FFT bins. So, it is important to convert the axis from bin sizes
% to range and doppler based on their Max values.

Mix=reshape(Mix,[Nr,Nd]);

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix,Nr,Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
% shift to zero component to the center of the array
sig_fft2 = fftshift (sig_fft2);

RDM = abs(sig_fft2);
RDM = 10*log10(RDM) ;


%% CFAR implementation

%Slide Window through the complete Range Doppler Map

% *%TODO* :
%Select the number of Training Cells in both the dimensions.
T_range = 8;
T_doppler = 8;

% *%TODO* :
%Select the number of Guard Cells in both dimensions around the Cell under 
%test (CUT) for accurate estimation
G_range = 4;
G_doppler = 4;

% Apply formulae from Udacitu lessons
Grid_size = (2*T_range + 2*G_range + 1)*(2*T_doppler + 2*G_doppler + 1);
Guard_cells = (2*G_doppler + 1)*(2*G_range + 1);
Training_cells = Grid_size - Guard_cells;
disp(Training_cells);
% *%TODO* :
% offset the threshold by SNR value in dB
offset = 6;

% *%TODO* :
%Create a vector to store noise_level for each iteration on training cells
noise_level = zeros(1,1);

% *%TODO* :
%design a loop such that it slides the CUT across range doppler map by
%giving margins at the edges for Training and Guard Cells.
%For every iteration sum the signal level within all the training
%cells. To sum convert the value from logarithmic to linear using db2pow
%function. Average the summed values for all of the training
%cells used. After averaging convert it back to logarithimic using pow2db.
%Further add the offset to it to determine the threshold. Next, compare the
%signal under CUT with this threshold. If the CUT level > threshold assign
%it a value of 1, else equate it to 0.

doppler_axis = linspace(-100,100,Nd);
range_axis = linspace(-200,200,Nr/2)*((Nr/2)/400);
figure,surf(doppler_axis,range_axis,RDM);

% Use RDM[x,y] as the matrix from the output of 2D FFT for implementing
% CFAR

% Create a new array - same dimensions are DBM in which we store the result
% of CFAR
cfar_output = zeros(Nr/2,Nd);

% loop over  Range - as we did before we take only half range of Nr
for i = 1: (Nr/2 -2*(T_range + G_range))
    
    % loop of doppler
    for j =1:(Nd- 2*(T_doppler + G_doppler))
     
        % Sum over Training + Guard cells
        s1 = sum(db2pow(RDM(i:i+2*T_range+2*G_range, j:j+2*T_doppler+2*G_doppler)),'all');
        
        % Sum over guard cells only
        s2 = sum(db2pow(RDM(i+T_range:i+T_range+2*G_range, j+T_doppler:j+T_doppler+2*G_doppler)),'all');
        
        % extract the noise from the guard cells (including CUT)
        noise_level = s1 - s2;
        
        threshold = noise_level/Training_cells;      
        threshold = pow2db(threshold) + offset;
        threshold = db2pow(threshold);
        
        % Now pick the cell under test which is the T+G cells away from the
        % first training cell and seasure the signal level
        signal = db2pow(RDM(i+T_range + G_range, j+T_doppler + G_doppler));
        
        %disp(signal);
        %disp(noise_level);
        % If the signal level at Cell Under Test belows the threshold then
        % assign it to 0 value
        if (signal <= threshold)
            signal = 0;
        else 
            signal = 1;
            disp("good");
        end
       
     cfar_output(i+T_range + G_range,j+T_doppler + G_doppler) = signal; 
     
        
    end
end


% *%TODO* :
% The process above will generate a thresholded block, which is smaller 
%than the Range Doppler Map as the CUT cannot be located at the edges of
%matrix. Hence,few cells will not be thresholded. To keep the map size same
% set those values to 0. 

% *%TODO* :
%display the CFAR output using the Surf function like we did for Range
%Doppler Response output.
figure,surf(doppler_axis,range_axis, cfar_output);
colorbar;



 
 