from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
    aws_lambda_event_sources as lambda_event_sources,
    aws_lambda as _lambda,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
    aws_sqs as sqs,
    aws_iam as iam,
    aws_secretsmanager as secretsmanager,
    aws_s3_notifications as s3n,
    aws_apigateway as apigw,
    CfnOutput
)
from constructs import Construct
from aje_cdk_libs.builders.resource_builder import ResourceBuilder
from aje_cdk_libs.models.configs import *
from aje_cdk_libs.constants.environments import Environments
from constants.paths import Paths
from constants.layers import Layers
import os
from dotenv import load_dotenv
import urllib.parse
from aje_cdk_libs.constants.project_config import ProjectConfig

class CdkAprendizajeGuiadoStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, project_config: ProjectConfig, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)         
        self.PROJECT_CONFIG = project_config        
        self.builder = ResourceBuilder(self, self.PROJECT_CONFIG)
        self.Paths = Paths(self.PROJECT_CONFIG.app_config)
        self.Layers = Layers(self.PROJECT_CONFIG.app_config, project_config.region_name, project_config.account_id)
 
        # Create all resources
        self.create_dynamodb_tables()
        # self.create_s3_buckets()
        self.create_lambda_layers()
        self.create_lambda_functions()
        self.create_api_gateway()
        self.create_outputs()
    
    def create_dynamodb_tables(self):
        """Create required DynamoDB tables"""
        # Case History Table
        dynamodb_config = DynamoDBConfig(
            table_name="case_history",
            partition_key="usuario_id",
            partition_key_type=dynamodb.AttributeType.NUMBER,
            sort_key="date_time",
            sort_key_type=dynamodb.AttributeType.STRING,
            removal_policy=RemovalPolicy.DESTROY
        )
        self.case_history_table = self.builder.build_dynamodb_table(dynamodb_config)
        
        # Evaluation History Table
        dynamodb_config = DynamoDBConfig(
            table_name="evaluation_history",
            partition_key="usuario_id",
            partition_key_type=dynamodb.AttributeType.NUMBER,
            sort_key="date_time",
            sort_key_type=dynamodb.AttributeType.STRING,
            removal_policy=RemovalPolicy.DESTROY
        )
        self.evaluation_history_table = self.builder.build_dynamodb_table(dynamodb_config)
        
        # Regenerated Challenges History Table
        dynamodb_config = DynamoDBConfig(
            table_name="regenerated_challenges_history",
            partition_key="usuario_id",
            partition_key_type=dynamodb.AttributeType.NUMBER,
            sort_key="date_time",
            sort_key_type=dynamodb.AttributeType.STRING,
            removal_policy=RemovalPolicy.DESTROY
        )
        self.regenerated_challenges_history_table = self.builder.build_dynamodb_table(dynamodb_config)
        
        # Learning Path History Table
        dynamodb_config = DynamoDBConfig(
            table_name="learning_path_history",
            partition_key="usuario_id",
            partition_key_type=dynamodb.AttributeType.NUMBER,
            sort_key="date_time",
            sort_key_type=dynamodb.AttributeType.STRING,
            removal_policy=RemovalPolicy.DESTROY
        )
        self.learning_path_history_table = self.builder.build_dynamodb_table(dynamodb_config)

    '''
    def create_s3_buckets(self):
        """Create S3 buckets for resource storage"""
        s3_config = S3Config(
            bucket_name="resources",
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL
        )
        self.resources_bucket = self.builder.build_s3_bucket(s3_config)
    '''
    
    def create_lambda_layers(self):
        """Create or reference required Lambda layers"""
        self.lambda_layer_powertools = _lambda.LayerVersion.from_layer_version_arn(
            self,
            "LambdaPowertoolsLayer",
            layer_version_arn=self.Layers.AWS_LAMBDA_LAYERS.get("layer_powertools")
        )
        
        self.lambda_layer_aje_libs = _lambda.LayerVersion.from_layer_version_arn(
            self,
            "LambdaAjeLibsLayer",
            layer_version_arn=self.Layers.AWS_LAMBDA_LAYERS.get("layer_aje_libs")
        )
        
        self.lambda_layer_pinecone = _lambda.LayerVersion.from_layer_version_arn(
            self,
            "LambdaPineconeLayer",
            layer_version_arn=self.Layers.AWS_LAMBDA_LAYERS.get("layer_pinecone")
        )
        
        self.lambda_layer_docs = _lambda.LayerVersion.from_layer_version_arn(
            self,
            "LambdaDocsLayer",
            layer_version_arn=self.Layers.AWS_LAMBDA_LAYERS.get("layer_docs")
        )
        
        self.lambda_layer_requests = _lambda.LayerVersion.from_layer_version_arn(
            self,
            "LambdaRequestsLayer",
            layer_version_arn=self.Layers.AWS_LAMBDA_LAYERS.get("layer_requests")
        )

    def create_lambda_functions(self):
        """Create all Lambda functions needed for the chatbot"""
        
        # Common environment variables for all Lambda functions
        common_env_vars = {
            "ENVIRONMENT": self.PROJECT_CONFIG.environment.value.lower(),
            "PROJECT_NAME": self.PROJECT_CONFIG.project_name,
            "OWNER": self.PROJECT_CONFIG.author,
            "DYNAMO_CASE_HISTORY_TABLE": self.case_history_table.table_name,
            "DYNAMO_EVALUATION_HISTORY_TABLE": self.evaluation_history_table.table_name,
            "DYNAMO_REGENERATED_CHALLENGES_HISTORY_TABLE": self.regenerated_challenges_history_table.table_name,
            "DYNAMO_LEARNING_PATH_HISTORY_TABLE": self.learning_path_history_table.table_name
        }
        
        # Create generar_ruta Lambda function
        function_name = "ruta-estandar-generar_ruta"
        handler_name = "generar_ruta"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{handler_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/ruta-estandar",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs, self.lambda_layer_pinecone]
        )
        self.ruta_estandar_generar_ruta_lambda = self.builder.build_lambda_function(lambda_config)

        # Create evaluar Lambda function
        function_name = "ruta-estandar-evaluar"
        handler_name = "evaluar"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{handler_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/ruta-estandar",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.ruta_estandar_evaluar_lambda = self.builder.build_lambda_function(lambda_config)
        
        # Create feedback Lambda function
        function_name = "ruta-estandar-feedback"
        handler_name = "feedback"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{handler_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/ruta-estandar",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.ruta_estandar_feedback_lambda = self.builder.build_lambda_function(lambda_config)
        
        # Create regenerar_reto Lambda function
        function_name = "ruta-estandar-regenerar_reto"
        handler_name = "regenerar_reto"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{handler_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/ruta-estandar",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.ruta_estandar_regenerar_reto_lambda = self.builder.build_lambda_function(lambda_config)


        # Create generar_caso Lambda function
        function_name = "metodo-caso-generar_caso"
        handler_name = "generar_caso"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{handler_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/metodo-caso",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.metodo_caso_generar_caso_lambda = self.builder.build_lambda_function(lambda_config)

        # Create generar_ruta Lambda function
        function_name = "metodo-caso-generar_ruta"
        handler_name = "generar_ruta"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{handler_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/metodo-caso",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.metodo_caso_generar_ruta_lambda = self.builder.build_lambda_function(lambda_config)

        # Create evaluar Lambda function
        function_name = "metodo-caso-evaluar"
        handler_name = "evaluar"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{handler_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/metodo-caso",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.metodo_caso_evaluar_lambda = self.builder.build_lambda_function(lambda_config)

        # Create feedback Lambda function
        function_name = "metodo-caso-feedback"
        handler_name = "feedback"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{handler_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/metodo-caso",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.metodo_caso_feedback_lambda = self.builder.build_lambda_function(lambda_config)

        
        # Grant permissions
        self.learning_path_history_table.grant_read_write_data(self.ruta_estandar_generar_ruta_lambda)
        self.learning_path_history_table.grant_read_write_data(self.metodo_caso_generar_ruta_lambda)

        self.evaluation_history_table.grant_read_write_data(self.ruta_estandar_evaluar_lambda)
        self.evaluation_history_table.grant_read_write_data(self.metodo_caso_evaluar_lambda)

        self.regenerated_challenges_history_table.grant_read_write_data(self.ruta_estandar_regenerar_reto_lambda)

        self.case_history_table.grant_read_write_data(self.metodo_caso_generar_caso_lambda)
        
        # Grant Bedrock permissions to Lambda functions
        bedrock_policy = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:Converse"
            ],
            resources=["*"]
        )
        
        # Grant Bedrock permissions to Lambda functions
        ssm_policy = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "ssm:GetParameter",
                "ssm:GetParameters"
            ],
            resources=["*"]
        )
        
        secrets_policy = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "secretsmanager:GetSecretValue"                
            ],
            resources=["*"]
        )
        
        self.ruta_estandar_generar_ruta_lambda.add_to_role_policy(bedrock_policy)
        self.ruta_estandar_evaluar_lambda.add_to_role_policy(bedrock_policy)
        self.ruta_estandar_feedback_lambda.add_to_role_policy(bedrock_policy)
        self.ruta_estandar_regenerar_reto_lambda.add_to_role_policy(bedrock_policy)
        self.metodo_caso_generar_caso_lambda.add_to_role_policy(bedrock_policy)
        self.metodo_caso_generar_ruta_lambda.add_to_role_policy(bedrock_policy)
        self.metodo_caso_evaluar_lambda.add_to_role_policy(bedrock_policy)
        self.metodo_caso_feedback_lambda.add_to_role_policy(bedrock_policy)
        
        self.ruta_estandar_generar_ruta_lambda.add_to_role_policy(ssm_policy)
        self.ruta_estandar_evaluar_lambda.add_to_role_policy(ssm_policy)
        self.ruta_estandar_feedback_lambda.add_to_role_policy(ssm_policy)
        self.ruta_estandar_regenerar_reto_lambda.add_to_role_policy(ssm_policy)
        self.metodo_caso_generar_caso_lambda.add_to_role_policy(ssm_policy)
        self.metodo_caso_generar_ruta_lambda.add_to_role_policy(ssm_policy)
        self.metodo_caso_evaluar_lambda.add_to_role_policy(ssm_policy)
        self.metodo_caso_feedback_lambda.add_to_role_policy(ssm_policy)

        self.ruta_estandar_generar_ruta_lambda.add_to_role_policy(secrets_policy)
        self.ruta_estandar_evaluar_lambda.add_to_role_policy(secrets_policy)
        self.ruta_estandar_feedback_lambda.add_to_role_policy(secrets_policy)
        self.ruta_estandar_regenerar_reto_lambda.add_to_role_policy(secrets_policy)
        self.metodo_caso_generar_caso_lambda.add_to_role_policy(secrets_policy)
        self.metodo_caso_generar_ruta_lambda.add_to_role_policy(secrets_policy)
        self.metodo_caso_evaluar_lambda.add_to_role_policy(secrets_policy)
        self.metodo_caso_feedback_lambda.add_to_role_policy(secrets_policy)
        
    def create_api_gateway(self):
        """
        Method to create the REST-API Gateway for exposing the chatbot
        functionalities.
        """ 
        # Create the API Gateway without specifying a default handler
        self.api_ruta_estandar = apigw.RestApi(
            self,
            f"{self.PROJECT_CONFIG.app_config['api_gw_name']}-{self.PROJECT_CONFIG.environment.value.lower()}",
            description=f"REST API Gateway for {self.PROJECT_CONFIG.project_name} in {self.PROJECT_CONFIG.environment.value} environment",
            deploy_options=apigw.StageOptions(
                stage_name=self.PROJECT_CONFIG.environment.value.lower(),
                description=f"REST API for {self.PROJECT_CONFIG.project_name}",
                metrics_enabled=True,
            ),    
            default_method_options=apigw.MethodOptions(
                api_key_required=False,
                authorization_type=apigw.AuthorizationType.NONE,
            ),
            endpoint_types=[apigw.EndpointType.REGIONAL],
            cloud_watch_role=False,
        )
        
        # Define REST-API resources
        root_agent_api = self.api_ruta_estandar.root.add_resource("api")
        root_agent_v1 = root_agent_api.add_resource("v1")

        # Endpoints for the main functionalities
        root_agent_generar_ruta_estandar = root_agent_v1.add_resource("generar_ruta_estandar")
        root_agent_evaluar_reto_estandar = root_agent_v1.add_resource("evaluar_reto_estandar")
        root_agent_feedback_estandar = root_agent_v1.add_resource("feedback_estandar")
        root_agent_regenerar_reto_estandar = root_agent_v1.add_resource("regenerar_reto_estandar")

        root_agent_generar_caso = root_agent_v1.add_resource("generar_caso")
        root_agent_generar_ruta_caso = root_agent_v1.add_resource("generar_ruta_caso")
        root_agent_evaluar_reto_caso = root_agent_v1.add_resource("evaluar_reto_caso")
        root_agent_feedback_caso = root_agent_v1.add_resource("feedback_caso")

        # Define all API-Lambda integrations for the API methods
        root_agent_generar_ruta_estandar.add_method("POST", apigw.LambdaIntegration(self.ruta_estandar_generar_ruta_lambda))
        root_agent_evaluar_reto_estandar.add_method("POST", apigw.LambdaIntegration(self.ruta_estandar_evaluar_lambda))
        root_agent_feedback_estandar.add_method("POST", apigw.LambdaIntegration(self.ruta_estandar_feedback_lambda))
        root_agent_regenerar_reto_estandar.add_method("POST", apigw.LambdaIntegration(self.ruta_estandar_regenerar_reto_lambda))

        root_agent_generar_caso.add_method("POST", apigw.LambdaIntegration(self.metodo_caso_generar_caso_lambda))
        root_agent_generar_ruta_caso.add_method("POST", apigw.LambdaIntegration(self.metodo_caso_generar_ruta_lambda))
        root_agent_evaluar_reto_caso.add_method("POST", apigw.LambdaIntegration(self.metodo_caso_evaluar_lambda))
        root_agent_feedback_caso.add_method("POST", apigw.LambdaIntegration(self.metodo_caso_feedback_lambda))
        
        # Store the deployment stage for use in outputs
        self.deployment_stage = self.PROJECT_CONFIG.environment.value.lower()
        
    def create_outputs(self):
        """Create CloudFormation outputs for important resources"""
        
        '''
        CfnOutput(self, "ResourcesBucketName", 
                value=self.resources_bucket.bucket_name,
                description="Resources S3 Bucket")
        '''
        
        CfnOutput(self, "ApiGatewayUrl", 
                value=f"https://{self.api_ruta_estandar.rest_api_id}.execute-api.{self.region}.amazonaws.com/{self.deployment_stage}/",
                description="API Gateway URL")
         