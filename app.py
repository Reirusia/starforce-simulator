import random
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from multiprocessing import Pool


success = 0.3
destroy = {15: 0.021, 16: 0.021, 17: 0.021, 18: 0.028, 19: 0.028, 20: 0.07, 21: 0.07}


def try_N(now, target, N = 100000, guard_destroy=False, sunday=False, starcatch=False):
    # result = []

    # for _ in range(N):
    #     r = try_once(now, target, guard_destroy, sunday, starcatch)
    #     result.append(r)
    with Pool() as p:
        result = p.map(try_once, [(now, target, guard_destroy, sunday, starcatch)]*N)
    

    df = pd.DataFrame(result, columns=['level', 'trial', 'guard_number', 'guard_level', 'star_succ_level', 'destroyed_level'], dtype=object)

    return df

def try_once(args):
    now, target, guard_destroy, sunday, starcatch = args

    trial = 0
    guard = 0
    guard_level_list = []
    star_succ_level_list = []
    while (True):
        trial += 1
        now, g, guard_level, star_succ, star_succ_level, destroyed_level = reinforce(now, guard_destroy, sunday,starcatch)

        # gaurded
        if g:
            guard += 1
            guard_level_list.append(guard_level)
        
        # success cause of starcatch
        if star_succ:
            star_succ_level_list.append(star_succ_level)

        # end
        if now == -1 or now == target:
            break

    if not guard_level_list:
        guard_level_list = None
    if not star_succ_level_list:
        star_succ_level_list = None

    return now, trial, guard, guard_level_list, star_succ_level_list, destroyed_level

def reinforce(now: int, guard_destroy: bool, sunday: bool, starcatch: bool):
    """
    reinforce

    Parameters
    ----------
    now : int
        level of item
    guard_destroy : bool
        guard destroy on 15, 16
    sunday : bool
        sunday event: 100% success on 15
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
    """

    level = now
    guard = False
    guard_level = None
    star_succ = False
    star_succ_level = None
    destroyed_level = None

    # sunday
    if sunday and now == 15:
        level = now + 1
        return level, guard, guard_level, star_succ, star_succ_level, destroyed_level
    
    d = dice()

    # starcatch
    adjusted_success = success
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
                level = now
            else:
                level = now -1
                
        else:  
            destroyed_level = now
            level = -1
    elif d >= 1 - adjusted_success:
        # success

        # success by starcatch
        if starcatch and not (d >= 1 - success):
            star_succ = True
            star_succ_level = now

        level = now + 1
    else:
        # failed
        if now == 15 or now == 20:
            level = now
        else:
            level = now - 1

    return level, guard, guard_level, star_succ, star_succ_level, destroyed_level
        
        
def dice():
    return random.random()

def calc(now, target, N, guard_destroy, sunday, starcatch):
    df = try_N(now, target, N, guard_destroy=guard_destroy, sunday=sunday, starcatch=starcatch)

    ###########
    # summary #
    ###########
    # success, destroy
    success_count = df.loc[df['level'] == target].shape[0]
    destroy_count = df.loc[df['level'] == -1].shape[0]
    success_rate = round(success_count / (success_count + destroy_count), 5) * 100
    destroy_rate = round(destroy_count / (success_count + destroy_count), 5) * 100

    # guard destroy
    guard_level_count = df['guard_number'].sum()

    # starcatch
    star_succ_level_values = df['star_succ_level'].value_counts()
    star_succ_level_count = 0
    for index, value in star_succ_level_values.items():
        star_succ_level_count += len(index) * value

    
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
            x=alt.X('X_label:N', axis=alt.Axis(grid=False)).title(title),
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


    #########################
    # destroyed level chart #
    #########################
    destroyed_levels = df['destroyed_level'].value_counts()
    destroyed_levels = destroyed_levels.reset_index()

    # add 15, 16 for using guard 
    for i in range(now, target):
        if i > 16:
            break
        if  i not in destroyed_levels['destroyed_level'].values:
            new_row = pd.DataFrame({'destroyed_level': [i], 'count': [0]})
            destroyed_levels = pd.concat([destroyed_levels, new_row], ignore_index=True)
    destroyed_level_chart = chart(destroyed_levels, 'destroyed_level', 'count', df.shape[0], '파괴횟수', True)


    ##############################
    # success by starcatch chart #
    ##############################
    # count now ~ target
    star_succ_level_values = df['star_succ_level'].value_counts()
    star_succ_level_dict = {}

    for i in range(15, target):
        star_succ_level_dict[f'{i}'] = 0

    for index, value in star_succ_level_values.items():
        for i in range(now, target):
            star_succ_level_dict[f'{i}'] += index.count(i) * value
    
    for i in range(15, target):
        if star_succ_level_dict[f'{i}'] == 0:
            del star_succ_level_dict[f'{i}']
    
    # Create DataFrame
    star_succ_level_df = pd.DataFrame(star_succ_level_dict, index=['count']).T.reset_index()
    starcatch_chart = chart(star_succ_level_df, 'index', 'count', df.shape[0], '스타캐치로 실패할 거 성공', True)


    #######################
    # guard destroy chart #
    #######################
    # count 15, 16
    guard_destroy_values = df['guard_level'].value_counts()
    count_15 = 0
    count_16 = 0
    for index, value in guard_destroy_values.items():
        count_15 += index.count(15) * value
        count_16 += index.count(16) * value

    # Create DataFrame
    guard_destroy_df = pd.DataFrame({'15': count_15, '16': count_16}, index=['count']).T.reset_index()
    guard_destroy_chart = chart(guard_destroy_df, 'index', 'count', df.shape[0], '파괴방지로 파괴될 거 방어', True)


    ###########
    # effects #
    ###########
    
    # Create DataFrame
    effects_df = pd.DataFrame({'파괴방지': df['guard_level'].count(), '스타캐치': df['star_succ_level'].count()}, index=['count']).T.reset_index()
    effect_chart = chart(effects_df, 'index', 'count', df.shape[0], '시도 중 한번 이상 효용을 봄')


    ##########
    # result #
    ##########
    # write
    st.write(f'달성률: **{round(success_rate, 3)}%**  파괴율: **{round(destroy_rate, 3)}%**  평균 강화횟수: **{round(df["trial"].mean(), 3)}회**')
    st.write(f'파괴방지로 방어한 파괴 횟수 평균: **{round(guard_level_count / N, 3)}회**')
    st.write(f'실패할거 스타캐치로 성공시킨 횟수 평균: **{round(star_succ_level_count / N, 3)}회**')
    st.write()
    st.write('> 파괴 또는 목표 달성 시까지를 1회라고 했을 때 그래프의 ( )는 1회당 발생하는 비율')

    # draw chart
    st.altair_chart(destroyed_level_chart, use_container_width=True)
    st.altair_chart(starcatch_chart, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(guard_destroy_chart, use_container_width=True)
    with col2:
        st.altair_chart(effect_chart, use_container_width=True)
    

def web():
    now = st.number_input('현재 스타포스(기본 15)', 15, 24, value=None)
    target = st.number_input('목표 스타포스(기본 22)', 16, 25, value=None)
    N = st.number_input('시행횟수(기본 100,000)', 1, 1000000, value=None, step=10000)

    starcatch = st.checkbox('스타캐치 사용', value=True)
    guard_destroy = st.checkbox('파괴방지', value=True)

    # sunday = st.radio(
    #     "썬데이 스타포스",
    #     ["없음", "10성 이하에서 강화시 1+1", "5, 10, 15성에서 성공확률 100%", "강화비용 30% 할인", "샤이닝 스타포스"])
    sunday = st.checkbox('5, 10, 15성에서 성공확률 100%')

    btn = st.button('계산하기')
    if btn:
        with st.spinner('Calculating'):
            if now == None:
                now = 15
            if target == None:
                target = 22
            if N == None:
                N = 100000
            calc(now, target, N, guard_destroy, sunday, starcatch)

if __name__ == '__main__':
    web()