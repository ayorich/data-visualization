import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from pyodide.http import pyfetch
import io

# Import the data
async def load_data():
    URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"
    resp = await pyfetch(URL)
    text = io.BytesIO((await resp.arrayBuffer()).to_py())
    df = pd.read_csv(text)
    return df

# TASK 1.1: Line chart showing automobile sales fluctuation from year to year
def task_1_1_line_chart(df):
    """
    Tree of Thought Reasoning:
    1. Need to group data by year and calculate average automobile sales
    2. Create a line plot with proper labels and title
    3. Add recession annotations for key years
    4. Include all years on x-axis with rotation for readability
    """
    # Create data for plotting
    df_line = df.groupby(df['Year'])['Automobile_Sales'].mean()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    df_line.plot(kind='line', marker='o', linewidth=2, markersize=6)
    
    # Customize the plot
    plt.xticks(list(range(1980, 2024)), rotation=75)
    plt.ylabel('Automobile Sales')
    plt.xlabel('Years')
    plt.title('Automobile Sales during Recession')
    
    # Add recession annotations
    plt.text(1982, 650, '1981-82 Recession', fontsize=10, ha='center')
    plt.text(1991, 650, '1991 Recession', fontsize=10, ha='center')
    plt.text(2001, 650, '2000-01 Recession', fontsize=10, ha='center')
    plt.text(2009, 650, '2008-09 Recession', fontsize=10, ha='center')
    plt.text(2020, 650, '2020 Covid-19', fontsize=10, ha='center')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('Line_Plot_1.png', dpi=300, bbox_inches='tight')
    plt.close()

# TASK 1.2: Multiple line chart for different vehicle types during recession
def task_1_2_vehicle_type_trends(df):
    """
    Tree of Thought Reasoning:
    1. Filter data for recession periods only
    2. Group by year and vehicle type, calculate average sales
    3. Normalize sales to compare trends across vehicle types
    4. Create multiple lines for each vehicle type
    5. Add recession year markers
    """
    # Filter recession data
    df_rec = df[df['Recession'] == 1]
    
    # Calculate average sales by year and vehicle type during recession
    df_Mline = df_rec.groupby(['Year', 'Vehicle_Type'], as_index=False)['Automobile_Sales'].mean()
    
    # Calculate normalized sales for better comparison
    df_Mline['Normalized_Sales'] = df_Mline.groupby('Vehicle_Type')['Automobile_Sales'].transform(lambda x: x / x.mean())
    
    # Set Year as index
    df_Mline.set_index('Year', inplace=True)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each vehicle type
    vehicle_types = df_Mline['Vehicle_Type'].unique()
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, vehicle_type in enumerate(vehicle_types):
        data = df_Mline[df_Mline['Vehicle_Type'] == vehicle_type]
        plt.plot(data.index, data['Normalized_Sales'], label=vehicle_type, 
                marker='o', linewidth=2, markersize=6, color=colors[i])
    
    # Highlight recession years
    recession_years = df_rec['Year'].unique()
    for year in recession_years:
        plt.axvline(x=year, color='gray', linestyle='--', alpha=0.5)
    
    # Customize plot
    plt.legend(title="Vehicle Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel("Normalized Sales")
    plt.xlabel("Year")
    plt.title("Normalized Automobile Sales by Vehicle Type During Recession")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('Line_Plot_2.png', dpi=300, bbox_inches='tight')
    plt.close()

# TASK 1.3: Seaborn bar chart comparing recession vs non-recession sales
def task_1_3_recession_comparison(df):
    """
    Tree of Thought Reasoning:
    1. Create grouped data for recession vs non-recession periods
    2. Use seaborn barplot for clean visualization
    3. Compare overall sales and vehicle-specific sales
    4. Add proper labels and title
    """
    # Overall comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Recession', y='Automobile_Sales', data=df, palette=['lightblue', 'lightcoral'])
    plt.xlabel('Period')
    plt.ylabel('Average Automobile Sales')
    plt.title('Average Automobile Sales during Recession and Non-Recession')
    plt.xticks(ticks=[0, 1], labels=['Non-Recession', 'Recession'])
    plt.show()
    
    # Vehicle type comparison
    grouped_df = df.groupby(['Recession', 'Vehicle_Type'])['Automobile_Sales'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Recession', y='Automobile_Sales', hue='Vehicle_Type', data=grouped_df)
    plt.xticks(ticks=[0, 1], labels=['Non-Recession', 'Recession'])
    plt.xlabel('Period')
    plt.ylabel('Average Automobile Sales')
    plt.title('Vehicle-Wise Sales during Recession and Non-Recession Period')
    plt.legend(title="Vehicle Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('Bar_Chart.png', dpi=300, bbox_inches='tight')
    plt.close()

# TASK 1.4: Subplotting GDP variation during recession and non-recession
def task_1_4_gdp_subplots(df):
    """
    Tree of Thought Reasoning:
    1. Split data into recession and non-recession periods
    2. Create two subplots using add_subplot
    3. Use seaborn lineplot for each subplot
    4. Add proper labels and titles
    """
    # Create dataframes for recession and non-recession period
    rec_data = df[df['Recession'] == 1]
    non_rec_data = df[df['Recession'] == 0]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 6))
    
    # Create different axes for subplotting
    ax0 = fig.add_subplot(1, 2, 1)  # add subplot 1 (1 row, 2 columns, first plot)
    ax1 = fig.add_subplot(1, 2, 2)  # add subplot 2 (1 row, 2 columns, second plot)
    
    # Plot 1: Recession period
    sns.lineplot(x='Year', y='GDP', data=rec_data, label='Recession', ax=ax0, marker='o')
    ax0.set_xlabel('Year')
    ax0.set_ylabel('GDP')
    ax0.set_title('GDP Variation during Recession Period')
    ax0.grid(True, alpha=0.3)
    
    # Plot 2: Non-recession period
    sns.lineplot(x='Year', y='GDP', data=non_rec_data, label='Non-Recession', ax=ax1, marker='o')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('GDP')
    ax1.set_title('GDP Variation during Non-Recession Period')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('Subplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# TASK 1.5: Bubble plot for seasonality impact
def task_1_5_seasonality_bubble(df):
    """
    Tree of Thought Reasoning:
    1. Filter data for non-recession periods only
    2. Use scatter plot with size parameter for bubble effect
    3. Use Month as x-axis, Automobile_Sales as y-axis
    4. Use Seasonality_Weight for bubble size
    5. Add proper labels and title
    """
    # Filter non-recession data
    non_rec_data = df[df['Recession'] == 0]
    
    # Create bubble plot
    plt.figure(figsize=(12, 8))
    
    # Use seasonality weight for bubble size
    size = non_rec_data['Seasonality_Weight'] * 1000  # Scale for visibility
    
    sns.scatterplot(data=non_rec_data, x='Month', y='Automobile_Sales', 
                   size=size, sizes=(50, 400), alpha=0.7)
    
    plt.xlabel('Month')
    plt.ylabel('Automobile Sales')
    plt.title('Seasonality impact on Automobile Sales')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('Bubble.png', dpi=300, bbox_inches='tight')
    plt.close()

# TASK 1.6: Scatter plots for correlation analysis
def task_1_6_correlation_scatter(df):
    """
    Tree of Thought Reasoning:
    1. Filter data for recession periods only
    2. Create scatter plot for Consumer Confidence vs Sales
    3. Create scatter plot for Price vs Sales
    4. Add trend lines if correlation exists
    5. Add proper labels and titles
    """
    # Filter recession data
    rec_data = df[df['Recession'] == 1]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Consumer Confidence vs Sales
    ax1.scatter(rec_data['Consumer_Confidence'], rec_data['Automobile_Sales'], alpha=0.7)
    ax1.set_xlabel('Consumer Confidence')
    ax1.set_ylabel('Automobile Sales')
    ax1.set_title('Consumer Confidence and Automobile Sales during Recessions')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(rec_data['Consumer_Confidence'], rec_data['Automobile_Sales'], 1)
    p = np.poly1d(z)
    ax1.plot(rec_data['Consumer_Confidence'], p(rec_data['Consumer_Confidence']), "r--", alpha=0.8)
    
    # Plot 2: Price vs Sales
    ax2.scatter(rec_data['Price'], rec_data['Automobile_Sales'], alpha=0.7)
    ax2.set_xlabel('Average Vehicle Price')
    ax2.set_ylabel('Automobile Sales')
    ax2.set_title('Relationship between Average Vehicle Price and Sales during Recessions')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(rec_data['Price'], rec_data['Automobile_Sales'], 1)
    p = np.poly1d(z)
    ax2.plot(rec_data['Price'], p(rec_data['Price']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('Scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

# TASK 1.7: Pie chart for advertising expenditure
def task_1_7_advertising_pie(df):
    """
    Tree of Thought Reasoning:
    1. Filter data for recession and non-recession periods
    2. Calculate total advertising expenditure for each period
    3. Create pie chart with proper labels and percentages
    4. Add title and customize colors
    """
    # Filter the data
    Rdata = df[df['Recession'] == 1]
    NRdata = df[df['Recession'] == 0]
    
    # Calculate the total advertising expenditure for both periods
    RAtotal = Rdata['Advertising_Expenditure'].sum()
    NRAtotal = NRdata['Advertising_Expenditure'].sum()
    
    # Create a pie chart for the advertising expenditure
    plt.figure(figsize=(10, 8))
    
    labels = ['Recession', 'Non-Recession']
    sizes = [RAtotal, NRAtotal]
    colors = ['lightcoral', 'lightblue']
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Advertising Expenditure during Recession and Non-Recession Periods')
    
    plt.show()
    
    # Save the plot
    plt.savefig('Pie_1.png', dpi=300, bbox_inches='tight')
    plt.close()

# TASK 1.8: Pie chart for vehicle type advertising expenditure during recession
def task_1_8_vehicle_advertising_pie(df):
    """
    Tree of Thought Reasoning:
    1. Filter data for recession periods only
    2. Group by vehicle type and sum advertising expenditure
    3. Create pie chart showing share of each vehicle type
    4. Add proper labels and percentages
    """
    # Filter the data for recession
    Rdata = df[df['Recession'] == 1]
    
    # Calculate the advertising expenditure by vehicle type during recessions
    VTexpenditure = Rdata.groupby('Vehicle_Type')['Advertising_Expenditure'].sum()
    
    # Create a pie chart for the share of each vehicle type in total expenditure during recessions
    plt.figure(figsize=(10, 8))
    
    labels = VTexpenditure.index
    sizes = VTexpenditure.values
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink']
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Share of Each Vehicle Type in Total Expenditure during Recessions')
    
    plt.show()
    
    # Save the plot
    plt.savefig('Pie_2.png', dpi=300, bbox_inches='tight')
    plt.close()

# TASK 1.9: Line plot for unemployment rate effect
def task_1_9_unemployment_line(df):
    """
    Tree of Thought Reasoning:
    1. Filter data for recession periods only
    2. Use seaborn lineplot with hue for vehicle types
    3. Plot unemployment rate vs sales
    4. Add proper labels and title
    """
    # Filter recession data
    df_rec = df[df['Recession'] == 1]
    
    # Create line plot
    plt.figure(figsize=(12, 8))
    
    sns.lineplot(data=df_rec, x='unemployment_rate', y='Automobile_Sales',
                hue='Vehicle_Type', style='Vehicle_Type', markers='o', err_style=None)
    
    plt.ylim(0, 850)
    plt.legend(loc=(0.05, 0.3))
    plt.xlabel('Unemployment Rate')
    plt.ylabel('Automobile Sales')
    plt.title('Effect of Unemployment Rate on Vehicle Type and Sales')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('line_plot_3.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main function to run all tasks
async def main():
    # Load data
    df = await load_data()
    print("Data loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Run all tasks
    print("\n=== TASK 1.1: Line chart for automobile sales fluctuation ===")
    task_1_1_line_chart(df)
    
    print("\n=== TASK 1.2: Vehicle type trends during recession ===")
    task_1_2_vehicle_type_trends(df)
    
    print("\n=== TASK 1.3: Recession vs non-recession comparison ===")
    task_1_3_recession_comparison(df)
    
    print("\n=== TASK 1.4: GDP variation subplots ===")
    task_1_4_gdp_subplots(df)
    
    print("\n=== TASK 1.5: Seasonality impact bubble plot ===")
    task_1_5_seasonality_bubble(df)
    
    print("\n=== TASK 1.6: Correlation scatter plots ===")
    task_1_6_correlation_scatter(df)
    
    print("\n=== TASK 1.7: Advertising expenditure pie chart ===")
    task_1_7_advertising_pie(df)
    
    print("\n=== TASK 1.8: Vehicle type advertising pie chart ===")
    task_1_8_vehicle_advertising_pie(df)
    
    print("\n=== TASK 1.9: Unemployment rate effect line plot ===")
    task_1_9_unemployment_line(df)
    
    print("\nAll visualizations completed and saved!")

# Run the main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 