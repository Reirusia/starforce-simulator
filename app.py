import random
from time import time
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
import streamlit as st
import altair as alt
from stqdm import stqdm
from multiprocessing import Pool


success = {i: round(0.95 - (0.05 * i), 2) for i in range(3)}
succ2 = {i: round(1 - (0.05 * i), 2) for i in range(3, 15)}
succ3 = {i: 0.3 for i in range(15, 22)}
for i in range(3):
    succ3[i+22] = 0.01 * (3 - i)

success.update(succ2)
success.update(succ3)

destroy = {i: 0 for i in range(15)}
destroy2 = {15: 0.021, 16: 0.021, 17: 0.021, 18: 0.028, 19: 0.028, 20: 0.07, 21: 0.07, 22:0.194, 23:0.294, 24:0.396}

destroy.update(destroy2)

def try_N(now, target, level, progress, N = 100000, spare_count=0, guard_destroy=False, succ_on_15=False, starcatch=False, discount_30p=False):
    with Pool() as p:
        result = p.map(try_once, [(now, target, level, spare_count, progress, N, guard_destroy, succ_on_15, starcatch, discount_30p)]*N)
    df = pd.DataFrame(result, columns=['level', 'trial', 'guard_number', 'guard_level', 'star_succ_level', 'destroyed_level', 'cost', 'used_spare'], dtype=object)

    return df

def try_once(args):
    n, target, level, spare_count, progress, N, guard_destroy, succ_on_15, starcatch, discount_30p = args

    trial = 0
    guard = 0
    guard_level_list = []
    star_succ_level_list = []
    cost = 0
    chance_time = 0
    used_spare = 0
    unlimited_spare = False
    if spare_count == -1:
        unlimited_spare = True

    while(True):
        now = n
        while (True):
            trial += 1
            now, g, guard_level, star_succ, star_succ_level, destroyed_level, c, d = reinforce(now, level, guard_destroy, succ_on_15,starcatch, discount_30p, chance_time)
            cost += c

            # gaurded
            if g:
                guard += 1
                guard_level_list.append(guard_level)
            
            # success cause of starcatch
            if star_succ:
                star_succ_level_list.append(star_succ_level)

            # chance time
            if d:
                chance_time += 1
            else:
                chance_time = 0

            # end
            if now == -1 or now == target:
                break

        if now == target:
            break   
        if unlimited_spare:
            used_spare += 1
            continue
        if spare_count <= used_spare:
            break
        else:
            used_spare += 1

    if not guard_level_list:
        guard_level_list = None
    if not star_succ_level_list:
        star_succ_level_list = None
    p = st.progress(0.0)
    p.progress

    return now, trial, guard, guard_level_list, star_succ_level_list, destroyed_level, cost, used_spare

def reinforce(now: int, level: int, guard_destroy: bool, succ_on_15: bool, starcatch: bool, discount_30p: bool, chance_time: int):
    """
    reinforce

    Parameters
    ----------
    now : int
        starforce level of item (0~ 24)
    level: int
        level of item
    guard_destroy : bool
        guard destroy on 15, 16
    succ_on_15 : bool
        succ_on_15 event: 100% success on 15
    starcatch: bool
        use starcatch

    Returns
    -------
    level: int
        result of trial. if -1, destroyed
    guard: bool
        guard destroy
    guard_level: int or None
        if guard destroy, record current level. if not, None
    star_succ: bool
        success cause of using starcatch
    star_succ_level: int or None
        if star_succ is True, record current level. if not, None
    destroyed_level: int or None
        if item destroyed, record current level. if not, None
    cost: int
        cost of reinforce
    """

    star_level = now
    guard = False
    guard_level = None
    star_succ = False
    star_succ_level = None
    destroyed_level = None
    decrease_star_level = False

    #calc cost
    if star_level < 10:
        cost = 1000 + ((level**3) * (star_level + 1)) / 25
    elif star_level == 10:
        cost = 1000 + ((level**3) * (star_level + 1)**2.7) / 400
    elif star_level == 11:
        cost = 1000 + ((level**3) * (star_level + 1)**2.7) / 220
    elif star_level == 12:
        cost = 1000 + ((level**3) * (star_level + 1)**2.7) / 150
    elif star_level == 13:
        cost = 1000 + ((level**3) * (star_level + 1)**2.7) / 110
    elif star_level == 14:
        cost = 1000 + ((level**3) * (star_level + 1)**2.7) / 75
    else:
        cost = 1000 + ((level**3) * (star_level + 1)**2.7) / 200

    cost = np.round(cost, -2)
    cost_result = cost
    if discount_30p:
        cost_result = np.round(cost * 0.7, -2)
    if guard_destroy and (star_level ==15 or star_level == 16):
        if not (succ_on_15 and star_level == 15) and not (chance_time == 2):
            cost_result += cost

    # chance time
    if chance_time == 2:
        star_level = now + 1
        return star_level, guard, guard_level, star_succ, star_succ_level, destroyed_level, cost_result, decrease_star_level
    
    # succ_on_15
    if succ_on_15:
        if star_level == 5 or star_level == 10 or star_level == 15:
            star_level = now + 1
            return star_level, guard, guard_level, star_succ, star_succ_level, destroyed_level, cost_result, decrease_star_level
    
    d = dice()

    # starcatch
    succ = success[now]
    adjusted_success = succ
    if starcatch:
        adjusted_success *= 1.05
    
    if d < destroy.get(now):
        # destroyed
        # guard_destroy
        if guard_destroy and (now == 15 or now ==16):
            # change to fail
            guard_level = now
            guard = True
            if now == 15:               
                star_level = now
            else:
                star_level = now -1
                decrease_star_level = True
                
        else:  
            destroyed_level = now
            star_level = -1
    elif d < adjusted_success + destroy.get(now):
        # success

        # success by starcatch
        if starcatch and not (d < succ + destroy.get(now)):
            star_succ = True
            star_succ_level = now

        star_level = now + 1
    else:
        # failed
        if now <= 15 or now == 20:
            star_level = now
        else:
            star_level = now - 1
            decrease_star_level = True

    return star_level, guard, guard_level, star_succ, star_succ_level, destroyed_level, cost_result, decrease_star_level
            
def dice():
    return random.random()

def format_number(number):
    if number >= 100000000:  # 1억 이상
        result = f'{number // 100000000}억'
        remainder = number % 100000000
        if remainder > 0:
            result += f'{remainder // 10000}만'
        return result
    elif number >= 10000:  # 1만 이상
        return f'{number // 10000}만'
    else:
        return str(number)
    
def calc(now, target, level, N, guard_destroy, succ_on_15, starcatch, discount_30p, spare_count, progress, debug=False):
    time_df_start = time()
    df = try_N(now, target, level, N, spare_count=spare_count, guard_destroy=guard_destroy, succ_on_15=succ_on_15, starcatch=starcatch, discount_30p=discount_30p)
    time_df_end = time()

    ###########
    # summary #
    ###########
    time_summary_start = time()
    # success, destroy
    success_count = df.loc[df['level'] == target].shape[0]
    destroy_count = df.loc[df['level'] == -1].shape[0]
    success_rate = round(success_count / (success_count + destroy_count), 5) * 100
    destroy_rate = round(destroy_count / (success_count + destroy_count), 5) * 100

    # cost
    cost_mean = int(round(df['cost'].mean(), -1))
    cost_median = int(round(df['cost'].median(), -1))
    mean_percentile = percentileofscore(df['cost'], cost_mean)

    # guard destroy
    guard_level_count = df['guard_number'].sum()

    # starcatch
    star_succ_level_values = df['star_succ_level'].value_counts()
    star_succ_level_count = 0
    for index, value in star_succ_level_values.items():
        star_succ_level_count += len(index) * value

    # success df
    success_df  = df[df['level'] == target]
    success_cost_mean = int(round(success_df['cost'].mean(), -1))
    success_cost_median = int(round(success_df['cost'].median(), -1))
    success_mean_percentile = percentileofscore(success_df['cost'], success_cost_mean)

    data = {
        'index': ['전체', '성공한 것만'],
        'colum': ['소모한 메소', '상위', '중간값(상위50%)'],
        'data': [[format_number(cost_mean), f'{mean_percentile:.3f}%', format_number(cost_median)],
              [format_number(success_cost_mean), f'{success_mean_percentile:.3f}%', format_number(success_cost_median)]],
    }

    table_df = pd.DataFrame(data['data'], columns=data['colum'], index=data['index'])
    
    time_summary_end = time()

    # write
    st.write(f'달성률: **{round(success_rate, 3)}%**  실패율: **{round(destroy_rate, 3)}%**')
    st.write(f'평균 강화횟수: **{round(df["trial"].mean(), 3)}회**  사용한 스페어 평균 개수: **{round(df["used_spare"].mean(), 3)}개**')
    st.write(f'파괴방지로 방어한 파괴 횟수 평균: **{round(guard_level_count / N, 3)}회**')
    st.write(f'실패할거 스타캐치로 성공시킨 횟수 평균: **{round(star_succ_level_count / N, 3)}회**')
    # st.write(f'소모한 평균 메소(전체): **{format_number(cost_mean)} 메소 [상위 {mean_percentile:.3f}%] (중간값: {format_number(cost_median)} 메소**)')
    # st.write(f'소모한 평균 메소(성공한 것만): **{format_number(success_cost_mean)} 메소 [상위 {success_mean_percentile:.3f}%] (중간값: {format_number(success_cost_median)} 메소**)')
    st.table(table_df)
    if debug:
        st.write(f'시뮬레이션에 걸린 시간: {time_df_end-time_df_start:.5f}')
        st.write(f'요약에 걸린 시간: {time_summary_end-time_summary_start:.5f}')
    st.write()
    st.write('> 아이템 전부 파괴 또는 목표 달성 시까지를 1회라고 했을 때 그래프의 ( )는 1회당 발생하는 비율')

    
    # chart function
    def chart(df, x_axis, y_axis, devide_by, title, parseInt=False):
        df['percentage'] = (df[y_axis] / devide_by) * 100
        df['percentage'] = df['percentage'].round(2)
        if parseInt:
            df['X_label'] = df.apply(lambda row: f"{int(row[x_axis])} ({row['percentage']:.2f}%)", axis=1)
        else:
            df['X_label'] = df.apply(lambda row: f"{row[x_axis]} ({row['percentage']:.2f}%)", axis=1)
        
        
        # Create bar chart
        c = alt.Chart(df).mark_bar(size=20).encode(
            x=alt.X('X_label:N', axis=alt.Axis(grid=False), sort = None).title(title),
            y='count',
            tooltip=[x_axis, 'count', 'percentage']
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14,
            labelAngle=0
        ).configure_view(
            strokeWidth=0
        )
        return c
    
    def base_position(now: int) -> int:
        if now >= 20:
            start_position = 20
        elif now >= 15:
            start_position = 15
        else:
            start_position = now
        return start_position
    

    ##############
    # cost chart #
    ##############
    time_cost_start = time()
    cost_df = df['cost'].reset_index()
    cost_df['cost'] /= 100000000
    min_cost = cost_df['cost'].min()
    max_cost = cost_df['cost'].max()
    cost_df.astype(str)
    
    # Create bar chart
    cost_chart = alt.Chart(cost_df).mark_bar(size=20).encode(
        x=alt.X('cost:Q', axis=alt.Axis(grid=False, labelAngle=0), bin=alt.Bin(extent=[min_cost, max_cost], maxbins=20),).title('스타포스 비용 (억)'),
        y=alt.Y('count():Q', title='count'),
        tooltip=['count()']
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    ).configure_view(
        strokeWidth=0
    )
    time_cost_end = time()


    #########################
    # destroyed level chart #
    #########################
    time_destroy_start = time()
    destroyed_levels = df['destroyed_level'].value_counts()
    destroyed_levels = destroyed_levels.reset_index()

    start_position = base_position(now)
    if start_position < 15:
        start_position = 15

    # add 15, 16 for using guard 
    for i in range(start_position, target):
        if i > 16:
            break
        if  i not in destroyed_levels['destroyed_level'].values:
            new_row = pd.DataFrame({'destroyed_level': [i], 'count': [0]})
            destroyed_levels = pd.concat([destroyed_levels, new_row], ignore_index=True)
    destroyed_levels.sort_values('destroyed_level', inplace=True)
    destroyed_level_chart = chart(destroyed_levels, 'destroyed_level', 'count', df.shape[0], '파괴횟수', True)
    time_destroy_end = time()


    ##############################
    # success by starcatch chart #
    ##############################
    # count now ~ target
    time_success_start = time()
    star_succ_level_values = df['star_succ_level'].value_counts()
    
    start_position = base_position(now)
    star_succ_level_dict = {i: 0 for i in range(start_position, target)}

    for index, value in star_succ_level_values.items():
        for i in range(start_position, target):
            star_succ_level_dict[i] += index.count(i) * value
    
    # Create DataFrame
    star_succ_level_df = pd.DataFrame(star_succ_level_dict, index=['count']).T.reset_index()
    starcatch_chart = chart(star_succ_level_df, 'index', 'count', df.shape[0], '스타캐치로 실패할 거 성공', True)
    time_success_end = time()


    #######################
    # guard destroy chart #
    #######################
    # count 15, 16
    time_guard_start = time()
    guard_destroy_values = df['guard_level'].value_counts()
    count_15 = 0
    count_16 = 0
    for index, value in guard_destroy_values.items():
        count_15 += index.count(15) * value
        count_16 += index.count(16) * value

    # Create DataFrame
    guard_destroy_df = pd.DataFrame({'15': count_15, '16': count_16}, index=['count']).T.reset_index()
    guard_destroy_chart = chart(guard_destroy_df, 'index', 'count', df.shape[0], '파괴방지로 파괴될 거 방어', True)
    time_guard_end = time()


    ###########
    # effects #
    ###########
    time_effect_start = time()
    # Create DataFrame
    effects_df = pd.DataFrame({'파괴방지': df['guard_level'].count(), '스타캐치': df['star_succ_level'].count()}, index=['count']).T.reset_index()
    effect_chart = chart(effects_df, 'index', 'count', df.shape[0], '시도 중 한번 이상 효용을 봄')
    time_effect_end = time()


    ##########
    # result #
    ##########
    time_draw_start = time()
    # draw chart
    st.altair_chart(cost_chart, use_container_width=True)
    st.altair_chart(destroyed_level_chart, use_container_width=True)
    st.altair_chart(starcatch_chart, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(guard_destroy_chart, use_container_width=True)
    with col2:
        st.altair_chart(effect_chart, use_container_width=True)
    time_draw_end = time()
    
    if debug:
        spent = [time_cost_end - time_cost_start, time_destroy_end - time_destroy_start, time_success_end - time_success_start, time_guard_end - time_guard_start, time_effect_end - time_effect_start, time_draw_end - time_draw_start]
        for i in spent:
            st.write(f'{i:.5f}')
    
def web():
    col1, col2 = st.columns(2)
    with col1:
        now = st.number_input('현재 스타포스(기본 15)', 0, 24, value=None, placeholder="15 (0 ~ 24)")
    with col2:    
        target = st.number_input('목표 스타포스(기본 22)', 1, 25, value=None, placeholder="22 (1 ~ 25)")
    col1, col2 = st.columns(2)
    with col1:
        level = st.number_input('장비 레벨(기본 160)', 0, 250, value=None, step=10, placeholder='160')
    with col2:    
        N = st.number_input('시행횟수(기본 50,000)', 1, 300000, value=None, step=10000, placeholder="50,000 (1 ~ 100,000)")
    
    spare = st.radio("스페어", ["없음", "무제한", "직접입력"], horizontal=True)

    if spare == "직접입력":
        spare_num = st.number_input("스페어 개수", 0, 100, value=None, step=1, placeholder='0 (0~100)')

    starcatch = st.checkbox('스타캐치 사용', value=False)
    guard_destroy = st.checkbox('파괴방지', value=False)

    sunday = st.radio(
        "썬데이 스타포스",
        ["없음", "5, 10, 15성에서 성공확률 100%", "강화비용 30% 할인", "샤이닝 스타포스"])

    btn = st.button('계산하기')
    if btn:
        progress = st.progress('please wait...')
        succ_on_15 = False
        discount_30p = False
        spare_count = 0
        if now == None:
            now = 15
        if target == None:
            target = 22
        if level == None:
            level = 160
        if N == None:
            N = 50000     
        if sunday == "5, 10, 15성에서 성공확률 100%":
            succ_on_15 = True
        elif sunday == "강화비용 30% 할인":
            discount_30p = True
        elif sunday == "샤이닝 스타포스":
            succ_on_15 = True
            discount_30p = True
        if spare == "없음":
            spare_count = 0
        elif spare == "무제한":
            spare_count = -1
        elif spare == "직접입력":
            if spare_num == None:
                spare_count = 0
            else:
                spare_count = spare_num

        calc(now, target, level, N, guard_destroy, succ_on_15, starcatch, discount_30p, spare_count, progress, False)

if __name__ == '__main__':
    web()