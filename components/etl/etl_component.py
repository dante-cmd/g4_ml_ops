from azure.ai.ml import command, Input, Output
from azure.ai.ml.entities import Component
from azure.ai.ml.constants import AssetTypes


def create_etl_component(environment, compute_target):
    """
    Creates an ETL component for data processing
    """
    etl_component = command(
        name="etl_component",
        display_name="ETL Component",
        description="Component for Extract, Transform, and Load operations",
        code="./src/data",
        command="python etl.py \
            --input_datastore ${{inputs.input_datastore}} \
            --output_datastore ${{outputs.output_datastore}} \
            --platinum_version ${{inputs.platinum_version}} \
            --ult_periodo ${{inputs.ult_periodo}} \
            --n_periodos ${{inputs.n_periodos}} \
            --all_periodos ${{inputs.all_periodos}}",
        inputs={
            "input_datastore": Input(type=AssetTypes.URI_FOLDER),
            "platinum_version": Input(type="string"),
            "ult_periodo": Input(type="integer"),
            "n_periodos": Input(type="string"),
            "all_periodos": Input(type="string")
        },
        outputs={
            "output_datastore": Output(type=AssetTypes.URI_FOLDER)
        },
        environment=environment,
    )

    return etl_component