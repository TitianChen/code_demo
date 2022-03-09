# Script to run future rainfall analaysis from archive datasets (UKCP18) 

conda activate Climate

model_collection="UKCP18"
emission_scenario="rcp85"
product="cpm_uk_2.2km"
resolution="daily"
area="UK"

products_path="/f/demo/cpm/"

echo "model_used: " ${model_collection} ${product}
echo "resolution: " ${resolution}

python samplecode_cpm_processing.py -h

echo "----------this processing task starts----------"
job_aim="-"
echo "Start processing the pre-processing"
echo "This pre-processing can be TIME CONSUMMING"
python samplecode_cpm_processing.py ${model_collection} ${emission_scenario} ${product} ${resolution} ${area} "${job_aim}" ${products_path} --recompute_from_beginning

echo "----------this processing task starts----------"
job_aim="rel_diff in mean rainfall"
echo "start processing:" "${job_aim}"
python samplecode_cpm_processing.py ${model_collection} ${emission_scenario} ${product} ${resolution} ${area} "${job_aim}" ${products_path} --plot_tag
echo "Results saved"

echo "----------this processing task starts----------"
job_aim="abs_diff in extreme rainfall"
echo "start processing:" "${job_aim}"
python samplecode_cpm_processing.py ${model_collection} ${emission_scenario} ${product} ${resolution} ${area} "${job_aim}" ${products_path} --plot_tag
echo "Results saved"

