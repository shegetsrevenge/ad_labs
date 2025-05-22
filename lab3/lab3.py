import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Налаштування конфігурації сторінки
st.set_page_config(layout="wide")

# Завантаження даних
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('full.csv')
        return data
    except FileNotFoundError:
        st.error("Файл 'full.csv' не знайдено. Переконайтеся, що файл знаходиться в тій же директорії, що й скрипт, або вкажіть правильний шлях.")
        return pd.DataFrame()

# Ініціалізація стану сесії для фільтрів
def initialize_session_state():
    if 'selected_index' not in st.session_state:
        st.session_state.selected_index = 'VHI'
    if 'selected_province' not in st.session_state:
        st.session_state.selected_province = 1
    if 'week_range' not in st.session_state:
        st.session_state.week_range = (1, 52)
    if 'year_range' not in st.session_state:
        st.session_state.year_range = (1982, 2024)
    if 'sort_ascending' not in st.session_state:
        st.session_state.sort_ascending = False
    if 'sort_descending' not in st.session_state:
        st.session_state.sort_descending = False

# Скидання фільтрів
def reset_filters():
    st.session_state.selected_index = 'VHI'
    st.session_state.selected_province = 1
    st.session_state.week_range = (1, 52)
    st.session_state.year_range = (1982, 2024)
    st.session_state.sort_ascending = False
    st.session_state.sort_descending = False

# Обробка та фільтрація даних
def filter_data(data, index, province, week_range, year_range, sort_ascending, sort_descending):
    filtered_data = data[
        (data['PROVINCE_ID'] == province) &
        (data['Week'].between(week_range[0], week_range[1])) &
        (data['Year'].between(year_range[0], year_range[1]))
    ]
    
    # Логіка сортування
    if sort_ascending and sort_descending:
        st.warning("Обрано сортування як за зростанням, так і за спаданням. Використовується сортування за замовчуванням (без сортування).")
    elif sort_ascending:
        filtered_data = filtered_data.sort_values(by=index, ascending=True)
    elif sort_descending:
        filtered_data = filtered_data.sort_values(by=index, ascending=False)
        
    return filtered_data

# Створення основного макета
col1, col2 = st.columns([1, 3])

# Бічна панель для інтерактивних елементів
with col1:
    st.header("Елементи керування")
    
    # Ініціалізація стану сесії
    initialize_session_state()
    
    # Випадаючий список для вибору індексу
    index_options = ['VCI', 'TCI', 'VHI']
    st.session_state.selected_index = st.selectbox(
        "Оберіть індекс",
        index_options,
        index=index_options.index(st.session_state.selected_index)
    )
    
    # Випадаючий список для вибору області
    data = load_data()
    if not data.empty:
        province_options = sorted(data['PROVINCE_ID'].unique())
        st.session_state.selected_province = st.selectbox(
            "Оберіть область",
            province_options,
            index=province_options.index(st.session_state.selected_province) if st.session_state.selected_province in province_options else 0
        )
    
    # Повзунок для вибору діапазону тижнів
    st.session_state.week_range = st.slider(
        "Оберіть діапазон тижнів",
        min_value=1,
        max_value=52,
        value=st.session_state.week_range
    )
    
    # Повзунок для вибору діапазону років
    st.session_state.year_range = st.slider(
        "Оберіть діапазон років",
        min_value=1982,
        max_value=2024,
        value=st.session_state.year_range
    )
    
    # Прапорці для сортування
    st.session_state.sort_ascending = st.checkbox("Сортувати за зростанням", value=st.session_state.sort_ascending)
    st.session_state.sort_descending = st.checkbox("Сортувати за спаданням", value=st.session_state.sort_descending)
    
    # Кнопка скидання фільтрів
    if st.button("Скинути фільтри"):
        reset_filters()

# Основна область для візуалізацій
with col2:
    st.header("Аналіз")
    
    # Перевірка, чи дані завантажено
    if data.empty:
        st.error("Немає даних для відображення. Перевірте наявність файлу 'full.csv'.")
    else:
        # Фільтрація даних на основі вибору
        filtered_data = filter_data(
            data,
            st.session_state.selected_index,
            st.session_state.selected_province,
            st.session_state.week_range,
            st.session_state.year_range,
            st.session_state.sort_ascending,
            st.session_state.sort_descending
        )
        
        # Створення вкладок
        tab1, tab2, tab3 = st.tabs(["Таблиця даних", "Графік часового ряду", "Порівняння областей"])
        
        # Вкладка 1: Відображення відфільтрованої таблиці
        with tab1:
            st.subheader("Відфільтровані дані")
            st.dataframe(filtered_data)
        
        # Вкладка 2: Графік часового ряду для обраної області
        with tab2:
            st.subheader(f"Часовий ряд {st.session_state.selected_index}")
            filtered_data['Date'] = filtered_data['Year'].astype(str) + '-Т' + filtered_data['Week'].astype(str)
            fig1 = px.line(
                filtered_data,
                x='Date',
                y=st.session_state.selected_index,
                title=f"{st.session_state.selected_index} для області {st.session_state.selected_province}",
                labels={'Date': 'Рік-Тиждень', st.session_state.selected_index: st.session_state.selected_index}
            )
            fig1.update_layout(
                xaxis_title="Рік-Тиждень",
                yaxis_title=st.session_state.selected_index,
                showlegend=True
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # Вкладка 3: Порівняльний графік між областями
        with tab3:
            st.subheader(f"Порівняння {st.session_state.selected_index} між областями")
            comparison_data = data[
                (data['Week'].between(st.session_state.week_range[0], st.session_state.week_range[1])) &
                (data['Year'].between(st.session_state.year_range[0], st.session_state.year_range[1]))
            ]
            
            # Агрегація даних за областями
            comparison_data = comparison_data.groupby(['PROVINCE_ID', 'Year', 'Week'])[st.session_state.selected_index].mean().reset_index()
            comparison_data['Date'] = comparison_data['Year'].astype(str) + '-Т' + comparison_data['Week'].astype(str)
            
            fig2 = go.Figure()
            for province in province_options:
                province_data = comparison_data[comparison_data['PROVINCE_ID'] == province]
                fig2.add_trace(
                    go.Scatter(
                        x=province_data['Date'],
                        y=province_data[st.session_state.selected_index],
                        name=f"Область {province}",
                        opacity=0.3 if province != st.session_state.selected_province else 1.0
                    )
                )
            
            fig2.update_layout(
                title=f"Порівняння {st.session_state.selected_index}",
                xaxis_title="Рік-Тиждень",
                yaxis_title=st.session_state.selected_index,
                showlegend=True
            )
            st.plotly_chart(fig2, use_container_width=True)

# Додавання цікавого факту
st.markdown("**Цікавий факт**: Дані показують, що індекс VHI (Індекс здоров’я рослинності) зазвичай досягає піку в середині року (тижні 25-30) у більшості областей, що, ймовірно, відповідає вегетаційному сезону, тоді як TCI (Індекс теплового стану) демонструє більшу мінливість, вказуючи на чутливість до змін температури.")