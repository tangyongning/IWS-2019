import pandas as pd

data_xls = pd.read_excel('Model 1 New Data.xls', 'Sheet1', index_col=None)
data_xls_2 = data_xls[['Date']]
data_xls = data_xls[['Total Solids', 'SS', 'BOD5', 'NH3', 'Org-N', 'P-TOT', 'SO4', 'TKN', 'PRCP_NOOA']]
data_xls = data_xls_2.join(data_xls)
data_xls.to_csv('Kirie_Edited_Data.csv', encoding='utf-8', index=False)

