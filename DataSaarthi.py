import pandas as pd
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db, auth
import google.generativeai as genai
import plotly.express as px
import base64
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats
def generate_discrepancy_report(source_df, target_df):
    
     #compares row-by row tabel updated by me on 3-02 Shoury
    report = ""
    report += f"Source CSV shape: {source_df.shape}, Target CSV shape: {target_df.shape}\n\n"
    source_cols = set(source_df.columns)
    target_cols = set(target_df.columns)
    missing_in_target = source_cols - target_cols
    extra_in_target = target_cols - source_cols

    if missing_in_target:
        report += f"Columns missing in target: {', '.join(missing_in_target)}\n"
    else:
        report += "No columns missing in target.\n"

    if extra_in_target:
        report += f"Extra columns in target not in source: {', '.join(extra_in_target)}\n"
    else:
        report += "No extra columns in target.\n"

    common_cols = source_cols.intersection(target_cols)
    min_rows = min(len(source_df), len(target_df))
    row_diffs = []
    for i in range(min_rows):
        differences = []
        for col in common_cols:
            source_val = source_df.iloc[i][col]
            target_val = target_df.iloc[i][col]
            if pd.isna(source_val) and pd.isna(target_val):
                continue
            if source_val != target_val:
                differences.append(
                    f"Row {i+1}, Column '{col}': source='{source_val}' vs target='{target_val}'"
                )
        if differences:
            row_diffs.extend(differences)
    if row_diffs:
        report += "\nDifferences in common rows:\n" + "\n".join(row_diffs)
    else:
        report += "\nNo differences found in common rows."

    if len(source_df) != len(target_df):
        report += f"\n\nRow count differs: Source has {len(source_df)} rows, Target has {len(target_df)} rows."

    return report

def create_pdf_report_download_link(report_text, filename="report.pdf"):
    """
    This f basic PDF out of it,
    HTML link 
    formatting pending Maitreya's task
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    text_object = c.beginText(40, height - 40)
    text_object.setFont("Helvetica", 10)
    for line in report_text.splitlines():
        text_object.textLine(line)
    c.drawText(text_object)
    c.showPage()
    c.save()
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF</a>'
    return href


def register_user(email, password):
    # register user as new
    try:
        user = auth.create_user(email=email, password=password)
        return {"status": "success", "message": f"User {email} registered successfully!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def login_user(email, password):
    # login
    ref = db.reference(f"users/{email.replace('.', ',')}").get()
    if ref and ref["password"] == password:
        return {"status": "success", "message": "Login successful!"}
    return {"status": "error", "message": "Invalid credentials."}

def save_user_to_db(email, password):
    ref = db.reference(f"users/{email.replace('.', ',')}")
    ref.set({"email": email, "password": password})


# Streamlit page

st.set_page_config(page_title="DataSaarthi: Data Analytics Platform", page_icon="📊", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "email" not in st.session_state:
    st.session_state["email"] = ""
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}
if "allow_second_upload" not in st.session_state:
    st.session_state["allow_second_upload"] = False
if "analysis_report" not in st.session_state:
    st.session_state["analysis_report"] = {}
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Firebase json open 
key_path = r"pandyashoury-firebase.json"
if not firebase_admin._apps:
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://pandyashoury-default-rtdb.firebaseio.com/"
    })

#  API key 1 of me account shourypandya05 
genai.configure(api_key="AIzaSyCAbmR8qWs_hlavvHNepmuKh33vBxwJj8A")


# CSS

st.markdown(
    """
    <style>
    body {
       background-color: #F7F7F8;
       color: #202123;
       font-family: Arial, sans-serif;
    }
    .main, .css-18e3th9 {
       background-color: #FFFFFF !important;
       color: #202123;
       border-radius: 8px;
       padding: 1em;
    }
    /* Main area inputs */
    .stTextInput > div > div > input {
       background-color: #FFFFFF;
       color: #202123;
       border: 1px solid #D0D0D0;
    }
    .stTextInput > div > label {
       color: #202123;
    }
    .stButton button {
       background-color: #007AFF;
       color: #FFFFFF;
       border: none;
       border-radius: 4px;
       padding: 0.4em 1em;
    }
    .stButton button:hover {
       background-color: #0051A8;
    }
    h1, h2, h3, h4, h5, h6 {
       color: #202123;
    }
    /* Sidebar styling with black background and blue accents */
    [data-testid="stSidebar"] {
       background: #000000;
       padding: 1em;
    }
    [data-testid="stSidebar"] .stTextInput input {
       background-color: #000000;
       color: #FFFFFF;
       border: 1px solid #007AFF;
    }
    [data-testid="stSidebar"] .stTextInput label {
       color: #007AFF;
    }
    [data-testid="stSidebar"] .stButton button {
       background-color: #007AFF;
       color: #FFFFFF;
       border: none;
       border-radius: 4px;
       padding: 0.4em 1em;
    }
    [data-testid="stSidebar"] .stButton button:hover {
       background-color: #0051A8;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
       color: #007AFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main header 

st.title("DataSaarthi: Data Analytics Platform")
st.markdown(
    """
<div style="background-color:#FFFFFF; padding:1em; border-radius:8px; border: 1px solid #D0D0D0;">
<h2 style="color:#202123;">Welcome to DataSaarthi</h2>
<p style="color:#202123;">
   Upload your CSV files to compare data, generate interactive charts, perform advanced analysis, and predict future trends.
</p>
</div>
""",
    unsafe_allow_html=True
)


# login and register stuff

with st.sidebar:
    st.markdown("## User Authentication")
    if not st.session_state["logged_in"]:
        st.subheader("Login")
        login_email = st.text_input("Email", key="login_email", help="Enter your registered email address")
        login_password = st.text_input("Password", type="password", key="login_password", help="Enter your password")
        if st.button("Login"):
            result = login_user(login_email, login_password)
            if result["status"] == "success":
                st.session_state["logged_in"] = True
                st.session_state["email"] = login_email
                st.success(result["message"])
            else:
                st.error(result["message"])

        st.markdown("---")
        st.subheader("Register")
        reg_email = st.text_input("New Email", key="reg_email", help="Enter a new email address")
        reg_password = st.text_input("New Password", type="password", key="reg_password", help="Choose a secure password")
        if st.button("Register"):
            result = register_user(reg_email, reg_password)
            if result["status"] == "success":
                save_user_to_db(reg_email, reg_password)
                st.success(result["message"])
            else:
                st.error(result["message"])
    else:
        st.success(f"Welcome, {st.session_state['email']}!")
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["email"] = ""
        st.markdown("---")

if st.session_state["logged_in"]:

    #  Upload Data 
    st.header("Data Upload and Preview")
    st.info("Step 1: Upload your primary dataset. Optionally, add a second dataset or a reference file.")
    
    col_upload, col_button = st.columns([3, 1])
    with col_upload:
        uploaded_file_1 = st.file_uploader(
            "Upload Your CSV File",
            type=["csv"],
            key="file_1",
            help="Choose a CSV file from your computer"
        )
        if uploaded_file_1:
            try:
                df1 = pd.read_csv(uploaded_file_1)
                st.session_state["datasets"]["Dataset 1"] = df1
                st.success("Dataset 1 uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading the dataset: {e}")

    with col_button:
        if "Dataset 1" in st.session_state["datasets"]:
            if st.button("Add Another Dataset", disabled=st.session_state["allow_second_upload"]):
                st.session_state["allow_second_upload"] = True

    if st.session_state["allow_second_upload"]:
        uploaded_file_2 = st.file_uploader(
            "Upload Second Dataset",
            type=["csv"],
            key="file_2",
            help="Choose a second CSV file if needed"
        )
        if uploaded_file_2:
            try:
                df2 = pd.read_csv(uploaded_file_2)
                st.session_state["datasets"]["Dataset 2"] = df2
                st.success("Second dataset uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading the second dataset: {e}")

    uploaded_reference_file = st.file_uploader(
        "Upload Reference CSV File (optional)",
        type=["csv"],
        key="reference_file",
        help="Compare your dataset(s) against a reference"
    )
    if uploaded_reference_file:
        try:
            df_reference = pd.read_csv(uploaded_reference_file)
            st.session_state["datasets"]["Reference"] = df_reference
            st.success("Reference dataset uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading the reference dataset: {e}")

    # edit data
    st.header("Data Preview and Editing")
    st.info("Step 2: Preview and directly edit your data below.")
    
    col_data1, col_data2 = st.columns(2)
    with col_data1:
        st.write("### Dataset 1")
        df1_editable = st.data_editor(
            st.session_state["datasets"]["Dataset 1"],
            num_rows="dynamic",
            use_container_width=True
        )
        st.session_state["datasets"]["Dataset 1"] = df1_editable
        st.write("#### Updated Preview:")
        st.dataframe(df1_editable)
        csv_data1 = df1_editable.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Edited Dataset 1 CSV",
            data=csv_data1,
            file_name="edited_dataset_1.csv",
            mime="text/csv"
        )

    with col_data2:
        if "Dataset 2" in st.session_state["datasets"]:
            st.write("### Dataset 2")
            df2_editable = st.data_editor(
                st.session_state["datasets"]["Dataset 2"],
                num_rows="dynamic",
                use_container_width=True
            )
            st.session_state["datasets"]["Dataset 2"] = df2_editable
            st.write("#### Updated Preview:")
            st.dataframe(df2_editable)
            csv_data2 = df2_editable.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Edited Dataset 2 CSV",
                data=csv_data2,
                file_name="edited_dataset_2.csv",
                mime="text/csv"
            )
        else:
            st.info("Upload a second dataset above to preview and edit.")

    # Step 3: Make Graph plot new changes done 14-04
    st.header("Interactive Graph Plotting")
    st.info("Select columns and a chart type to generate your interactive plot. You may also ask the AI for suggestions.")
    df1 = st.session_state["datasets"]["Dataset 1"]
    available_columns = df1.columns.tolist()
    if not available_columns:
        st.warning("Dataset 1 has no columns to plot. Please upload a valid CSV.")
    else:
        x_axis_column = st.selectbox("Select X-axis column", available_columns, help="Choose a column for the X axis")
        y_axis_column = st.selectbox("Select Y-axis column", available_columns, help="Choose a column for the Y axis")
        chart_types = ["Scatter", "Line", "Bar", "Histogram", "Pie", "3D Scatter"]
        selected_chart_type = st.selectbox("Select Chart Type", chart_types, help="Pick a chart type")
        if st.button("Get AI Suggestion for Chart Type"):
            prompt_text = f"""
            We have two columns: {x_axis_column} and {y_axis_column}.
            Their data types are:
            - {x_axis_column}: {df1[x_axis_column].dtype}
            - {y_axis_column}: {df1[y_axis_column].dtype}
            From the following options: {chart_types}, suggest the best chart type and provide a one-sentence reason.
            """
            ai_response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt_text)
            if ai_response and hasattr(ai_response, 'text'):
                st.write("**AI Suggestion:**", ai_response.text)
            else:
                st.error("No response received from the model.")
        if st.button("Generate Plot"):
            try:
                if selected_chart_type == "Scatter":
                    fig = px.scatter(df1, x=x_axis_column, y=y_axis_column,
                                     title=f"{y_axis_column} vs {x_axis_column} (Scatter)",
                                     labels={x_axis_column: x_axis_column, y_axis_column: y_axis_column})
                elif selected_chart_type == "Line":
                    fig = px.line(df1, x=x_axis_column, y=y_axis_column,
                                  title=f"{y_axis_column} vs {x_axis_column} (Line)",
                                  labels={x_axis_column: x_axis_column, y_axis_column: y_axis_column})
                elif selected_chart_type == "Bar":
                    fig = px.bar(df1, x=x_axis_column, y=y_axis_column,
                                 title=f"{y_axis_column} vs {x_axis_column} (Bar)",
                                 labels={x_axis_column: x_axis_column, y_axis_column: y_axis_column})
                elif selected_chart_type == "Histogram":
                    fig = px.histogram(df1, x=x_axis_column,
                                       title=f"Histogram of {x_axis_column}",
                                       labels={x_axis_column: x_axis_column})
                elif selected_chart_type == "Pie":
                    fig = px.pie(df1, names=x_axis_column,
                                 title=f"Pie Chart of {x_axis_column}")
                elif selected_chart_type == "3D Scatter":
                    st.write("**Choose your Z-axis column for the 3D scatter**")
                    z_axis_column = st.selectbox("Select Z-axis column", available_columns)
                    fig = px.scatter_3d(df1, x=x_axis_column, y=y_axis_column, z=z_axis_column,
                                        color=x_axis_column,
                                        title=f"3D Scatter Plot: {x_axis_column}, {y_axis_column}, {z_axis_column}")
                else:
                    st.warning("Unsupported chart type selected.")
                    fig = None

                if fig is not None:
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating the plot: {e}")

    # 3D Graph
    st.header("3D Graph")
    st.info("Select columns and choose a 3D graph type to dynamically explore your data with AI insights.")
    df1 = st.session_state["datasets"]["Dataset 1"]
    available_columns = df1.columns.tolist()
    if available_columns:
        x_axis = st.selectbox("Select X-axis column", available_columns, key="3d_x")
        y_axis = st.selectbox("Select Y-axis column", available_columns, key="3d_y")
        z_axis = st.selectbox("Select Z-axis column", available_columns, key="3d_z")
        graph_type = st.selectbox("Select Graph Type", 
                                  ["3D Scatter Plot", "3D Surface Plot", "3D Line Plot"], 
                                  key="3d_graph_type")
        if st.button("Generate 3D Graph", key="generate_3d"):
            try:
                if graph_type == "3D Scatter Plot":
                    fig3d = px.scatter_3d(df1, x=x_axis, y=y_axis, z=z_axis,
                                          title=f"3D Scatter: {x_axis} vs {y_axis} vs {z_axis}",
                                          color=x_axis)
                elif graph_type == "3D Surface Plot":
                    fig3d = px.scatter_3d(df1, x=x_axis, y=y_axis, z=z_axis,
                                          title=f"3D Surface (Sampled) Plot: {x_axis} vs {y_axis} vs {z_axis}")
                elif graph_type == "3D Line Plot":
                    fig3d = px.line_3d(df1, x=x_axis, y=y_axis, z=z_axis,
                                       title=f"3D Line Plot: {x_axis} vs {y_axis} vs {z_axis}")
                else:
                    st.warning("Unsupported graph type.")
                    fig3d = None

                if fig3d is not None:
                    fig3d.update_layout(template="plotly_white")
                    st.plotly_chart(fig3d, use_container_width=True)
                    st.success("Interactive 3D graph generated successfully.")
            except Exception as e:
                st.error(f"Error generating 3D plot: {e}")
    else:
        st.warning("Dataset 1 does not contain any columns for 3D plotting.")

    # Analysis
    st.header("Data Analysis and Insights")
    st.info("Utilize AI-powered analysis and advanced insights.")
    dataset_names = [name for name in st.session_state["datasets"].keys() if name != "Reference"]

    if len(dataset_names) == 2:
        st.subheader("Combined Analysis (Merging Two Datasets)")
        if st.button("Combine & Analyze All Data"):
            df1 = st.session_state["datasets"][dataset_names[0]]
            df2 = st.session_state["datasets"][dataset_names[1]]
            combined_data = pd.concat([df1, df2], ignore_index=True).to_dict(orient='records')
            response_combined = genai.GenerativeModel("gemini-1.5-flash").generate_content(
                f"Analyze these combined rows: {combined_data}. Provide insights or anomalies found."
            )
            if response_combined and hasattr(response_combined, 'text'):
                st.markdown(f"### Combined Analysis Report\n{response_combined.text}")
                st.session_state["analysis_report"]["combined"] = response_combined.text
            else:
                st.error("No response received from the model.")
        if "combined" in st.session_state["analysis_report"]:
            report_text = st.session_state["analysis_report"]["combined"]
            if st.download_button(
                label="Download Combined Analysis as CSV",
                data=pd.DataFrame([{"report": report_text}]).to_csv(index=False),
                file_name="combined_analysis_report.csv",
                mime="text/csv"
            ):
                pass
            pdf_link = create_pdf_report_download_link(report_text, filename="combined_analysis_report.pdf")
            st.markdown(pdf_link, unsafe_allow_html=True)

    st.subheader("Individual Dataset Analysis")
    for ds_name in dataset_names:
        st.write(f"**Analyze {ds_name}**")
        if st.button(f"Analyze {ds_name}"):
            df = st.session_state["datasets"][ds_name]
            query_dataset = df.to_dict(orient="records")
            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
                f"Analyze this dataset: {query_dataset}. Provide anomalies or interesting insights."
            )
            if response and hasattr(response, 'text'):
                st.markdown(f"### Analysis Report for {ds_name}\n{response.text}")
                st.session_state["analysis_report"][ds_name] = response.text
            else:
                st.error("No response from the AI model.")
        if ds_name in st.session_state["analysis_report"]:
            report_text = st.session_state["analysis_report"][ds_name]
            if st.download_button(
                label=f"Download {ds_name} Report as CSV",
                data=pd.DataFrame([{"report": report_text}]).to_csv(index=False),
                file_name=f"{ds_name}_analysis_report.csv",
                mime="text/csv"
            ):
                pass
            pdf_link = create_pdf_report_download_link(report_text, filename=f"{ds_name}_analysis_report.pdf")
            st.markdown(pdf_link, unsafe_allow_html=True)

    # Step 5: Compare (my idea ) worked on 15-05 completed 
    if "Reference" in st.session_state["datasets"]:
        st.header("Reference Comparison")
        st.info("Compare your dataset(s) against the reference CSV to identify discrepancies.")
        reference_df = st.session_state["datasets"]["Reference"]
        for ds_name in dataset_names:
            if st.button(f"Compare {ds_name} with Reference"):
                ds_df = st.session_state["datasets"][ds_name]
                report_ref = generate_discrepancy_report(reference_df, ds_df)
                st.markdown(f"### Discrepancy Report for {ds_name} vs Reference")
                rows = [line for line in report_ref.split("\n") if line.strip() != ""]
                report_df = pd.DataFrame(rows, columns=["Comparison"])
                st.table(report_df)
                if st.download_button(
                    label=f"Download {ds_name} vs Reference Report as CSV",
                    data=pd.DataFrame([{"report": report_ref}]).to_csv(index=False),
                    file_name=f"{ds_name}_reference_discrepancy_report.csv",
                    mime="text/csv"
                ):
                    pass
                pdf_link = create_pdf_report_download_link(report_ref, filename=f"{ds_name}_reference_discrepancy_report.pdf")
                st.markdown(pdf_link, unsafe_allow_html=True)

    # Talk with CSV  updated on 11-03 by me 
    st.header("Talk with CSV")
    st.info("Interactively query your CSV data by selecting a range of rows and entering your question.")
    df1 = st.session_state["datasets"]["Dataset 1"]
    colA, colB = st.columns(2)
    with colA:
        start = st.number_input("Start Row (for query)", min_value=0, max_value=len(df1)-1 if len(df1)>0 else 0, value=0)
    with colB:
        end = st.number_input("End Row (for query)", min_value=0, max_value=len(df1), value=len(df1))
    user_query = st.chat_input("Enter your query about the selected rows:")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if user_query:
        st.session_state["messages"].append({"role": "user", "content": user_query})
        selected_data = df1.iloc[start:end].to_dict(orient="records")
        prompt_with_data = f"Data from row {start} to {end}:\n{selected_data}\nUser Query: {user_query}"
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt_with_data)
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            if response and hasattr(response, 'text'):
                st.markdown(response.text)
                st.session_state["messages"].append({"role": "assistant", "content": response.text})
            else:
                st.error("No response received from the model.")

