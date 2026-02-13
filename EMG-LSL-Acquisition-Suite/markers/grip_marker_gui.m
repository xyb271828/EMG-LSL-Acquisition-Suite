function run_grip_experiment_gui
%=========================================
% 握力实验标注器（GUI + LSL Markers）
% 30 次握力：握 2s + 休 2s
% 与 Daq_revised_4ch.py（EMG->LSL）+ LabRecorder 配套
%
% Marker 设计（建议）：
%   REST_k        : 第 k 次前的休息段开始
%   GRIP_START_k  : 第 k 次握力开始
%   GRIP_END_k    : 第 k 次握力结束
%   EXP_START / EXP_END
%=========================================

    %% ====== LSL 初始化 ======
    disp('LSLライブラリを読み込んでいます…');
    lib = lsl_loadlib();

    unique_id = char(java.util.UUID.randomUUID());
    info = lsl_streaminfo(lib, 'GripMarkers', 'Markers', 1, 0, 'cf_string', unique_id);
    outlet = lsl_outlet(info);
    fprintf('LSL Markerストリームを作成しました: GripMarkers (%s)\n', unique_id);

    %% ====== 实验参数 ======
    N_TRIALS   = 40;
    GRIP_SEC  = 2;   % 每次握力持续
    REST_SEC  = 2;   % 每次间隔休息（握力之间）
    COUNTDOWN_FPS = 10; % 倒计时刷新频率（越大越丝滑）

    %% ====== GUI ======
    fig = uifigure('Name', '握力実験マーカーGUI (GripMarkers)', 'Position', [500 300 520 320]);

    lbl_title = uilabel(fig, ...
        'Text', '準備完了：開始後、画面の指示に従って握ってください', ...
        'FontSize', 18, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', ...
        'Position', [30 240 460 50]);

    lbl_phase = uilabel(fig, ...
        'Text', 'フェーズ：-', ...
        'FontSize', 14, ...
        'HorizontalAlignment', 'center', ...
        'Position', [30 200 460 30]);

    lbl_trial = uilabel(fig, ...
        'Text', sprintf('Trial：0/%d', N_TRIALS), ...
        'FontSize', 14, ...
        'HorizontalAlignment', 'center', ...
        'Position', [30 170 460 30]);

    lbl_time = uilabel(fig, ...
        'Text', '残り時間：-', ...
        'FontSize', 14, ...
        'HorizontalAlignment', 'center', ...
        'Position', [30 140 460 30]);

    lbl_hint = uilabel(fig, ...
        'Text', 'ヒント：LabRecorderで「EMG」と「GripMarkers」の両方を選択してください', ...
        'FontSize', 12, ...
        'HorizontalAlignment', 'center', ...
        'Position', [30 110 460 25]);

    startBtn = uibutton(fig, 'push', ...
        'Text', '実験開始', ...
        'FontSize', 16, ...
        'BackgroundColor', [0.2 0.6 1], ...
        'Position', [100 35 140 55], ...
        'ButtonPushedFcn', @(btn,event) start_experiment());

    stopBtn = uibutton(fig, 'push', ...
        'Text', '緊急停止', ...
        'FontSize', 16, ...
        'BackgroundColor', [1 0.4 0.4], ...
        'Position', [280 35 140 55], ...
        'Enable', 'off', ...
        'ButtonPushedFcn', @(btn,event) stop_experiment());

    % 运行状态
    is_running = false;
    abort_flag = false;

    %% ====== 实验逻辑 ======
    function start_experiment()
        if is_running
            return;
        end
        is_running = true;
        abort_flag = false;

        startBtn.Enable = 'off';
        stopBtn.Enable  = 'on';

        % 实验开始 marker
        push_marker('EXP_START');

        lbl_title.Text = '実験を開始しました！';
        drawnow;

        % （可选）开始前给 2 秒准备
        show_countdown('準備', 2);

        for k = 1:N_TRIALS
            if abort_flag
                break;
            end

            lbl_trial.Text = sprintf('Trial：%d/%d', k, N_TRIALS);

            % ---- 休息段 ----
            lbl_title.Text = 'リラックスしてください（力を入れない）';
            lbl_phase.Text = sprintf('フェーズ：REST（%ds）', REST_SEC);
            push_marker(sprintf('REST_%02d', k));
            show_countdown('REST', REST_SEC);

            if abort_flag
                break;
            end

            % ---- 握力段 ----
            lbl_title.Text = '握ってください！';
            lbl_phase.Text = sprintf('フェーズ：GRIP（%ds）', GRIP_SEC);
            push_marker(sprintf('GRIP_START_%02d', k));
            show_countdown('GRIP', GRIP_SEC);
            push_marker(sprintf('GRIP_END_%02d', k));
        end

        % 实验结束 marker
        if abort_flag
            push_marker('EXP_ABORT');
            lbl_title.Text = '緊急停止しました（EXP_ABORT）';
        else
            push_marker('EXP_END');
            lbl_title.Text = '実験終了！（EXP_END）';
        end

        lbl_phase.Text = 'フェーズ：-';
        lbl_time.Text  = '残り時間：-';
        stopBtn.Enable = 'off';
        startBtn.Text  = '完了';
        startBtn.BackgroundColor = [0.6 0.6 0.6];
        is_running = false;
    end

    function stop_experiment()
        abort_flag = true;
        stopBtn.Enable = 'off';
        lbl_title.Text = '停止処理中…';
        drawnow;
    end

    function push_marker(marker_str)
        % marker_str 必须是 cellstr
        outlet.push_sample({marker_str});
        % 你想在命令行也看到日志的话，可以打开：
        % fprintf('[%s] marker: %s\n', datestr(now,'HH:MM:SS.FFF'), marker_str);
    end

    function show_countdown(tag, total_sec)
        % 用更细的步长刷新倒计时，减少 pause(1) 带来的不准
        dt = 1 / COUNTDOWN_FPS;
        t_left = total_sec;

        while t_left > 0
            if abort_flag
                return;
            end
            lbl_time.Text = sprintf('残り時間：%.1f 秒', t_left);
            drawnow;
            pause(dt);
            t_left = t_left - dt;
        end

        lbl_time.Text = sprintf('残り時間：0.0 秒');
        drawnow;
    end
end

