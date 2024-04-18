#=========================================================================================
# feature_importance.py ------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Plot the feature importance of the training inputs for the BDT -------------------------
#-----------------------------------------------------------------------------------------

import xgboost as xgb
import matplotlib
matplotlib.use('Agg') # prevents opening displays (fast), must use before pyplot
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS") # CMS plot style

# Load the trained model from a saved file
#model = xgb.Booster()  # Create a blank model
model = xgb.XGBClassifier()  # Create a blank model
#model.load_model('../models/svjbdt_Aug01_allsignals_qcdttjets.json')  # Load the saved model
#model.load_model('models/svjbdt_Nov08_allsignals_iterative_qcdttjets.json')  # Load the saved model
#model.load_model('models/svjbdt_Nov08_allsignals_iterative_qcdttjets.json')  # Load the saved model
model.load_model('models/svjbdt_Feb28_lowmass_iterative_qcdtt_100p38.json')  # Load the saved model

# Define the training features list
training_features = [
    'girth', 'ptd', 'axismajor', 'axisminor',
    'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi',
    'ak15_chad_ef', 'ak15_nhad_ef', 'ak15_elect_ef', 'ak15_muon_ef', 'ak15_photon_ef',
]

# Add feature names to the model
model.feature_names = training_features


'''
# Get feature importance scores
importance = model.get_score(importance_type='weight')

# Sort the importance scores by their values
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Extract sorted feature names and their corresponding importance values
sorted_features, sorted_values = zip(*sorted_importance)

'''

# Get feature importances from the model
importance_values = model.feature_importances_

# Sort features and their importance values
sorted_features = [feature for _, feature in sorted(zip(importance_values, training_features), reverse=True)]
sorted_importance_values = sorted(importance_values, reverse=True)

# Plot feature importance
plt.barh(range(len(sorted_features)), sorted_importance_values, align='center')
plt.yticks(range(len(sorted_features)), sorted_features)
plt.xlabel('Importance')
plt.ylabel('Feature')
#plt.title('Feature Importance')

# Options to make the plot fancier 
hep.cms.label(rlabel="2018 (13 TeV)")

# Plot feature importance
#xgb.plot_importance(model)
#xgb.plot_importance(model, feature_names = training_features)
#plt.yticks(range(len(training_features)), sorted_features)  # Set y-axis labels
#plt.yticks(range(len(training_features)), sorted_features)  # Set y-axis labels
plt.savefig("plots/feature_importance.png", bbox_inches='tight', pad_inches=1.0)
plt.savefig("plots/feature_importance.pdf", bbox_inches='tight', pad_inches=1.0)



