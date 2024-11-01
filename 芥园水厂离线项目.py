import numpy as np
import pandas as pd
import pickle
import streamlit as st

from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title('芥园水厂智慧加药项目')

with st.container(border=True):

    st.header('1. 模型训练')

    with st.container(border=True):
        st.subheader('1.1 上传训练数据')
        uploaded_file = st.file_uploader("请根据**注意事项**中的要求上传Excel数据")
        with st.expander('注意事项'):
            st.write('1111111')

    with st.container(border=True):
        st.subheader('1.2 加载训练数据')
        if uploaded_file is not None:
            dataframe = pd.read_excel(uploaded_file)
            dataframe['时间'] = pd.to_datetime(dataframe['时间'], format='%Y-%m-%d %H:%M:%S')
            dataframe.index = dataframe['时间']
            dataframe.drop(columns=['时间'], axis=1, inplace=True)
            st.write(dataframe)
        else:
            st.write('请上传数据')

    with st.container(border=True):

        st.subheader('1.3 训练模型')

        if uploaded_file is not None:
            if st.button("点击开始训练，若不满意则可重复点击再次进行训练", use_container_width=True):
                data = pd.read_excel(uploaded_file)
                # 数据处理
                data = data.dropna()
                y = data[['次氯酸钠', '铁盐', '铝盐', '硫酸铵']]
                y_names = y.columns
                X = data.drop(['时间', '次氯酸钠', '铁盐', '铝盐', '硫酸铵'], axis=1)
                X_names = X.columns
                y_na = y['次氯酸钠']
                y_fe = y['铁盐']
                y_al = y['铝盐']
                y_nh = y['硫酸铵']
                # 数据归一化
                scaler = preprocessing.MinMaxScaler()
                scaled_X = scaler.fit_transform(X)
                scaled_X = pd.DataFrame(scaled_X)
                # NaClO预测
                y_na = pd.DataFrame(y_na)
                scaled_y_na = scaler.fit_transform(y_na)
                scaled_y_na = pd.DataFrame(scaled_y_na)
                scaled_data_na = pd.concat([scaled_X, scaled_y_na], axis=1)

                na_name = ['次氯酸钠']
                na_data_names = pd.concat([pd.Series(X_names), pd.Series(na_name)], axis=0)
                scaled_data_na.columns = na_data_names

                data_train_na, data_test_na = train_test_split(scaled_data_na, test_size=0.2)
                y_train_na = data_train_na[na_name]
                X_train_na = data_train_na.drop(na_name, axis=1)
                y_test_na = data_test_na[na_name]
                X_test_na = data_test_na.drop(na_name, axis=1)

                model_na = RandomForestRegressor().fit(X_train_na, y_train_na)
                y_pred_na = model_na.predict(X_test_na)
                y_test_na = scaler.inverse_transform(y_test_na)
                y_pred_na = scaler.inverse_transform(y_pred_na.reshape(-1, 1))
                mse_test_na = metrics.mean_squared_error(y_test_na, y_pred_na)
                rmse_test_na = np.sqrt(mse_test_na)
                rmse_test_na = "%.2f" % rmse_test_na
                st.write(f'训练完成！模型预测次氯酸钠投加量的RMSE为：{rmse_test_na} mg/L')

                with open('model_na.pkl', 'wb') as f:
                    pickle.dump(model_na, f)

                # FeCl3预测
                y_fe = pd.DataFrame(y_fe)
                scaled_y_fe = scaler.fit_transform(y_fe)
                scaled_y_fe = pd.DataFrame(scaled_y_fe)
                scaled_data_fe = pd.concat([scaled_X, scaled_y_fe], axis=1)

                fe_name = ['铁盐']
                fe_data_names = pd.concat([pd.Series(X_names), pd.Series(fe_name)], axis=0)
                scaled_data_fe.columns = fe_data_names

                data_train_fe, data_test_fe = train_test_split(scaled_data_fe, test_size=0.2)
                y_train_fe = data_train_fe[fe_name]
                X_train_fe = data_train_fe.drop(fe_name, axis=1)
                y_test_fe = data_test_fe[fe_name]
                X_test_fe = data_test_fe.drop(fe_name, axis=1)

                model_fe = RandomForestRegressor().fit(X_train_fe, y_train_fe)
                y_pred_fe = model_fe.predict(X_test_fe)
                y_test_fe = scaler.inverse_transform(y_test_fe)
                y_pred_fe = scaler.inverse_transform(y_pred_fe.reshape(-1, 1))
                mse_test_fe = metrics.mean_squared_error(y_test_fe, y_pred_fe)
                rmse_test_fe = np.sqrt(mse_test_fe)
                rmse_test_fe = "%.2f" % rmse_test_fe
                st.write(f'训练完成！模型预测铁盐投加量的RMSE为：{rmse_test_fe} mg/L')

                with open('model_fe.pkl', 'wb') as f:
                    pickle.dump(model_fe, f)
                # PAC预测
                y_al = pd.DataFrame(y_al)
                scaled_y_al = scaler.fit_transform(y_al)
                scaled_y_al = pd.DataFrame(scaled_y_al)
                scaled_data_al = pd.concat([scaled_X, scaled_y_al], axis=1)

                al_name = ['铝盐']
                al_data_names = pd.concat([pd.Series(X_names), pd.Series(al_name)], axis=0)
                scaled_data_al.columns = al_data_names

                data_train_al, data_test_al = train_test_split(scaled_data_al, test_size=0.2)
                y_train_al = data_train_al[al_name]
                X_train_al = data_train_al.drop(al_name, axis=1)
                y_test_al = data_test_al[al_name]
                X_test_al = data_test_al.drop(al_name, axis=1)

                model_al = RandomForestRegressor().fit(X_train_al, y_train_al)
                y_pred_al = model_al.predict(X_test_al)
                y_test_al = scaler.inverse_transform(y_test_al)
                y_pred_al = scaler.inverse_transform(y_pred_al.reshape(-1, 1))
                mse_test_al = metrics.mean_squared_error(y_test_al, y_pred_al)
                rmse_test_al = np.sqrt(mse_test_al)
                rmse_test_al = "%.2f" % rmse_test_al
                st.write(f'训练完成！模型预测铝盐投加量的RMSE为：{rmse_test_al} mg/L')
                with open('model_al.pkl', 'wb') as f:
                    pickle.dump(model_al, f)
                # NH4SO2预测
                y_nh = pd.DataFrame(y_nh)
                scaled_y_nh = scaler.fit_transform(y_nh)
                scaled_y_nh = pd.DataFrame(scaled_y_nh)
                scaled_data_nh = pd.concat([scaled_X, scaled_y_nh], axis=1)

                nh_name = ['硫酸铵']
                nh_data_names = pd.concat([pd.Series(X_names), pd.Series(nh_name)], axis=0)
                scaled_data_nh.columns = nh_data_names

                data_train_nh, data_test_nh = train_test_split(scaled_data_nh, test_size=0.2)
                y_train_nh = data_train_nh[nh_name]
                X_train_nh = data_train_nh.drop(nh_name, axis=1)
                y_test_nh = data_test_nh[nh_name]
                X_test_nh = data_test_nh.drop(nh_name, axis=1)

                model_nh = RandomForestRegressor().fit(X_train_nh, y_train_nh)
                y_pred_nh = model_nh.predict(X_test_nh)
                y_test_nh = scaler.inverse_transform(y_test_nh)
                y_pred_nh = scaler.inverse_transform(y_pred_nh.reshape(-1, 1))
                mse_test_nh = metrics.mean_squared_error(y_test_nh, y_pred_nh)
                rmse_test_nh = np.sqrt(mse_test_nh)
                rmse_test_nh = "%.2f" % rmse_test_nh
                st.write(f'训练完成！模型预测硫酸铵投加量的RMSE为：{rmse_test_nh} mg/L')
                with open('model_nh.pkl', 'wb') as f:
                    pickle.dump(model_nh, f)

        else:
            st.write('请上传数据')

with st.container(border=True):
    st.header('2. 模型预测')
    with st.form('my_form'):
        st.subheader('2.1 输入数据')
        st.write('请按要求输入相关信息，然后按预测键。')
        FLOW = st.number_input('进水量', min_value=300.0, max_value=500.0, step=0.01)
        IT = st.number_input('进水浊度', min_value=0.1, max_value=25.0, step=0.01)
        AN = st.number_input('氨氮', min_value=0.01, max_value=0.3, step=0.01)
        pH = st.number_input('pH', min_value=7.0, max_value=9.0, step=0.01)
        ALK = st.number_input('碱度', min_value=70, max_value=120, step=1)
        TEM = st.number_input('水温', min_value=2.0, max_value=32.0, step=0.1)
        ALG = st.number_input('藻类', min_value=5, max_value=2200, step=1)
        CHL = st.number_input('叶绿素', min_value=0.0, max_value=20.0, step=0.01)
        RAL = st.number_input('预期余铝', min_value=0.0, max_value=0.2, step=0.01)
        ET = st.number_input('预期出水浊度', min_value=0.0, max_value=1.0, step=0.01)
        
        submitted = st.form_submit_button('预测')
    with st.container(border=True):
        st.subheader('2.2 投药量预测')
        if submitted:
            input_data = pd.DataFrame({'进水量': [FLOW], '浑浊度': [IT],
                                       '氨氮': [AN], 'pH': [pH], '总碱度': [ALK],
                                       '水温': [TEM], '藻类': [ALG],
                                       '叶绿素': [CHL], '余铝': [RAL],
                                       '出水浊度': [ET]})
            def norm(x, xmin, xmax):
                x = (x - xmin)/(xmax-xmin)
                return x
            data = pd.read_excel(uploaded_file)
            input_data['进水量'] = norm(input_data['进水量'], data['进水量'].min(), data['进水量'].max())
            input_data['浑浊度'] = norm(input_data['浑浊度'], data['浑浊度'].min(), data['浑浊度'].max())
            input_data['氨氮'] = norm(input_data['氨氮'], data['氨氮'].min(), data['氨氮'].max())
            input_data['pH'] = norm(input_data['pH'], data['pH'].min(), data['pH'].max())
            input_data['总碱度'] = norm(input_data['总碱度'], data['总碱度'].min(), data['总碱度'].max())
            input_data['水温'] = norm(input_data['水温'], data['水温'].min(), data['水温'].max())
            input_data['藻类'] = norm(input_data['藻类'], data['藻类'].min(), data['藻类'].max())
            input_data['叶绿素'] = norm(input_data['叶绿素'], data['叶绿素'].min(), data['叶绿素'].max())
            input_data['余铝'] = norm(input_data['余铝'], data['余铝'].min(), data['余铝'].max())
            input_data['出水浊度'] = norm(input_data['出水浊度'], data['出水浊度'].min(), data['出水浊度'].max())

            with open('model_na.pkl', 'rb') as f:
                model_na = pickle.load(f)
            prediction_na = model_na.predict(input_data)
            prediction_na = prediction_na * (data['次氯酸钠'].max() - data['次氯酸钠'].min()) + data['次氯酸钠'].min() - 0.5
            prediction_na = "%.2f" % prediction_na
            st.subheader(f'预测需要投加 {prediction_na} mg/L的次氯酸钠！')

            with open('model_fe.pkl', 'rb') as f:
                model_fe = pickle.load(f)
            prediction_fe = model_fe.predict(input_data)
            prediction_fe = prediction_fe * (data['铁盐'].max() - data['铁盐'].min()) + data['铁盐'].min() - 0.5
            prediction_fe = "%.2f" % prediction_fe
            st.subheader(f'预测需要投加 {prediction_fe} mg/L的铁盐！')

            with open('model_al.pkl', 'rb') as f:
                model_al = pickle.load(f)
            prediction_al = model_al.predict(input_data)
            prediction_al = prediction_al * (data['铁盐'].max() - data['铁盐'].min()) + data['铁盐'].min() - 0.5
            prediction_al = "%.2f" % prediction_al
            st.subheader(f'预测需要投加 {prediction_al} mg/L的铝盐！')

            with open('model_nh.pkl', 'rb') as f:
                model_nh = pickle.load(f)

            prediction_nh = model_nh.predict(input_data)
            prediction_nh = prediction_nh * (data['硫酸铵'].max() - data['硫酸铵'].min()) + data['硫酸铵'].min()
            prediction_nh = "%.2f" % prediction_nh
            st.subheader(f'预测需要投加 {prediction_nh} mg/L的硫酸铵！')

        else:
            st.subheader('请完成以上所有步骤!')