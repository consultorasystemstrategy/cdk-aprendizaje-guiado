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

class CdkAgentGuidedStack(Stack):
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

        dynamodb_config = DynamoDBConfig(
            table_name="case_history",
            partition_key="usuario_id",
            partition_key_type=dynamodb.AttributeType.NUMBER,
            sort_key="date_time",
            sort_key_type=dynamodb.AttributeType.STRING,
            removal_policy=RemovalPolicy.DESTROY
        )
        self.case_history_table = self.builder.build_dynamodb_table(dynamodb_config)

        dynamodb_config = DynamoDBConfig(
            table_name="learning_path_history",
            partition_key="usuario_id",
            partition_key_type=dynamodb.AttributeType.NUMBER,
            sort_key="date_time",
            sort_key_type=dynamodb.AttributeType.STRING,
            removal_policy=RemovalPolicy.DESTROY
        )
        self.learning_path_history_table = self.builder.build_dynamodb_table(dynamodb_config)

        dynamodb_config = DynamoDBConfig(
            table_name="regenerated_challenges_history",
            partition_key="usuario_id",
            partition_key_type=dynamodb.AttributeType.NUMBER,
            sort_key="date_time",
            sort_key_type=dynamodb.AttributeType.STRING,
            removal_policy=RemovalPolicy.DESTROY
        )
        self.regenerated_challenges_history_table = self.builder.build_dynamodb_table(dynamodb_config)
        
        dynamodb_config = DynamoDBConfig(
            table_name="evaluation_history",
            partition_key="usuario_id",
            partition_key_type=dynamodb.AttributeType.NUMBER,
            sort_key="date_time",
            sort_key_type=dynamodb.AttributeType.STRING,
            removal_policy=RemovalPolicy.DESTROY
        )
        self.evaluation_history_table = self.builder.build_dynamodb_table(dynamodb_config)
        
        dynamodb_config = DynamoDBConfig(
            table_name="challenge_evaluation_history",
            partition_key="usuario_id",
            partition_key_type=dynamodb.AttributeType.NUMBER,
            sort_key="reto_iteracion_id",
            sort_key_type=dynamodb.AttributeType.STRING,
            removal_policy=RemovalPolicy.DESTROY
        )
        self.challenge_evaluation_history_table = self.builder.build_dynamodb_table(dynamodb_config)

        dynamodb_config = DynamoDBConfig(
            table_name="challenge_feedback_history",
            partition_key="usuario_id",
            partition_key_type=dynamodb.AttributeType.NUMBER,
            sort_key="reto_resultado_id",
            sort_key_type=dynamodb.AttributeType.STRING,
            removal_policy=RemovalPolicy.DESTROY
        )
        self.challenge_feedback_history_table = self.builder.build_dynamodb_table(dynamodb_config)

        dynamodb_config = DynamoDBConfig(
            table_name="challenge_enrichment_history",
            partition_key="history_id",
            partition_key_type=dynamodb.AttributeType.STRING,
            sort_key="date_time",
            sort_key_type=dynamodb.AttributeType.STRING,
            removal_policy=RemovalPolicy.DESTROY
        )
        self.challenge_enrichment_history_table = self.builder.build_dynamodb_table(dynamodb_config)

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
        
        # Common environment variables for all Lambda functions
        common_env_vars = {
            "ENVIRONMENT": self.PROJECT_CONFIG.environment.value.lower(),
            "ENTERPRISE": self.PROJECT_CONFIG.enterprise,
            "PROJECT_NAME": self.PROJECT_CONFIG.project_name,
            "OWNER": self.PROJECT_CONFIG.author,
            "CASE_HISTORY_TABLE": self.case_history_table.table_name,
            "LEARNING_PATH_HISTORY_TABLE": self.learning_path_history_table.table_name,
            "REGENERATED_CHALLENGES_HISTORY_TABLE": self.regenerated_challenges_history_table.table_name,
            "EVALUATION_HISTORY_TABLE": self.evaluation_history_table.table_name,
            "CHALLENGE_EVALUATION_HISTORY_TABLE": self.challenge_evaluation_history_table.table_name, # Nueva tabla para evaluaciones de retos
            "CHALLENGE_FEEDBACK_HISTORY_TABLE": self.challenge_feedback_history_table.table_name,
            "CHALLENGE_ENRICHMENT_HISTORY_TABLE": self.challenge_enrichment_history_table.table_name
        }
        
        # Ruta estándar
        function_name = "path_generate"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/learning-path/v1",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs, self.lambda_layer_pinecone]
        )
        self.path_generate_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "path_regenerate_challenge"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/learning-path/v1",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.path_regenerate_challenge_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "path_evaluate"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/learning-path/v1",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.path_evaluate_lambda = self.builder.build_lambda_function(lambda_config)
        
        function_name = "path_feedback"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/learning-path/v1",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.path_feedback_lambda = self.builder.build_lambda_function(lambda_config)

        # Ruta con caso
        function_name = "case_generate"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/case/v1",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.case_generate_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "case_generate_path"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/case/v1",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.case_generate_path_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "case_evaluate"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/case/v1",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.case_evaluate_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "case_feedback"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/case/v1",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.case_feedback_lambda = self.builder.build_lambda_function(lambda_config)

        # Ruta estándar v2
        function_name = "path_generate_worker"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/learning-path/v2/path_generate_v2",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs, self.lambda_layer_pinecone]
        )
        self.path_generate_worker_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "path_generate_start"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/learning-path/v2/path_generate_v2",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs, self.lambda_layer_pinecone]
        )
        self.path_generate_start_lambda = self.builder.build_lambda_function(lambda_config)
        self.path_generate_start_lambda.add_environment("PATH_GENERATE_WORKER_LAMBDA", self.path_generate_worker_lambda.function_name)

        function_name = "path_regenerate_challenge_v2"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/learning-path/v2",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.path_regenerate_challenge_v2_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "path_evaluate_v2"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/learning-path/v2",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.path_evaluate_v2_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "path_feedback_v2"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/learning-path/v2",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.path_feedback_v2_lambda = self.builder.build_lambda_function(lambda_config)

        # Ruta con caso v2
        function_name = "case_generate_worker"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/case/v2/case_generate_v2",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.case_generate_worker_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "case_generate_start"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/case/v2/case_generate_v2",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.case_generate_start_lambda = self.builder.build_lambda_function(lambda_config)
        self.case_generate_start_lambda.add_environment("CASE_GENERATE_WORKER_LAMBDA", self.case_generate_worker_lambda.function_name)

        function_name = "case_generate_path_worker"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/case/v2/case_generate_path_v2",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.case_generate_path_worker_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "case_generate_path_start"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/case/v2/case_generate_path_v2",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.case_generate_path_start_lambda = self.builder.build_lambda_function(lambda_config)
        self.case_generate_path_start_lambda.add_environment("CASE_GENERATE_PATH_WORKER_LAMBDA", self.case_generate_path_worker_lambda.function_name)

        function_name = "case_evaluate_v2"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/case/v2",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.case_evaluate_v2_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "case_feedback_v2"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/case/v2",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.case_feedback_v2_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "case_enrichment_challenge_worker"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/case/v2/case_enrichment_challenge",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.case_enrichment_challenge_worker_lambda = self.builder.build_lambda_function(lambda_config)

        function_name = "case_enrichment_challenge_start"
        lambda_config = LambdaConfig(
            function_name=function_name,
            handler=f"{function_name}/lambda_function.lambda_handler",
            code_path=f"{self.Paths.LOCAL_ARTIFACTS_LAMBDA_CODE}/case/v2/case_enrichment_challenge",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(30),
            environment=common_env_vars,
            layers=[self.lambda_layer_powertools, self.lambda_layer_aje_libs]
        )
        self.case_enrichment_challenge_start_lambda = self.builder.build_lambda_function(lambda_config)
        self.case_enrichment_challenge_start_lambda.add_environment("CASE_ENRICHMENT_CHALLENGE_WORKER_LAMBDA", self.case_enrichment_challenge_worker_lambda.function_name)

        # Grant DynamoDB permissions to Lambda functions
        self.learning_path_history_table.grant_read_write_data(self.path_generate_lambda)
        self.learning_path_history_table.grant_read_write_data(self.case_generate_path_lambda)
        self.learning_path_history_table.grant_read_write_data(self.path_generate_worker_lambda) # Ruta estándar v2
        self.learning_path_history_table.grant_read_write_data(self.case_generate_path_worker_lambda) # Ruta con caso v2

        self.regenerated_challenges_history_table.grant_read_write_data(self.path_regenerate_challenge_lambda)
        self.regenerated_challenges_history_table.grant_read_write_data(self.path_regenerate_challenge_v2_lambda) # Ruta estándar v2

        self.evaluation_history_table.grant_read_write_data(self.path_evaluate_lambda)
        self.evaluation_history_table.grant_read_write_data(self.case_evaluate_lambda)
        self.challenge_evaluation_history_table.grant_read_write_data(self.path_evaluate_v2_lambda) # Ruta estándar v2
        self.challenge_evaluation_history_table.grant_read_write_data(self.case_evaluate_v2_lambda) # Ruta con caso v2

        self.challenge_feedback_history_table.grant_read_write_data(self.path_feedback_v2_lambda)
        self.challenge_feedback_history_table.grant_read_write_data(self.case_feedback_v2_lambda)

        self.case_history_table.grant_read_write_data(self.case_generate_lambda)
        self.case_history_table.grant_read_write_data(self.case_generate_worker_lambda)

        self.challenge_enrichment_history_table.grant_read_write_data(self.case_enrichment_challenge_worker_lambda)

        # Grant invocation permissions to the worker Lambda from the start Lambda
        self.path_generate_worker_lambda.grant_invoke(self.path_generate_start_lambda)
        self.case_generate_worker_lambda.grant_invoke(self.case_generate_start_lambda)
        self.case_generate_path_worker_lambda.grant_invoke(self.case_generate_path_start_lambda)
        self.case_enrichment_challenge_worker_lambda.grant_invoke(self.case_enrichment_challenge_start_lambda)

        bedrock_policy = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:Converse"
            ],
            resources=["*"]
        )
        
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
        
        self.path_generate_lambda.add_to_role_policy(bedrock_policy)
        self.path_generate_worker_lambda.add_to_role_policy(bedrock_policy)
        self.path_evaluate_lambda.add_to_role_policy(bedrock_policy)
        self.path_evaluate_v2_lambda.add_to_role_policy(bedrock_policy)
        self.path_feedback_lambda.add_to_role_policy(bedrock_policy)
        self.path_feedback_v2_lambda.add_to_role_policy(bedrock_policy)
        self.path_regenerate_challenge_lambda.add_to_role_policy(bedrock_policy)
        self.path_regenerate_challenge_v2_lambda.add_to_role_policy(bedrock_policy)
        self.case_generate_lambda.add_to_role_policy(bedrock_policy)
        self.case_generate_worker_lambda.add_to_role_policy(bedrock_policy)
        self.case_generate_path_lambda.add_to_role_policy(bedrock_policy)
        self.case_generate_path_worker_lambda.add_to_role_policy(bedrock_policy)
        self.case_evaluate_lambda.add_to_role_policy(bedrock_policy)
        self.case_evaluate_v2_lambda.add_to_role_policy(bedrock_policy)
        self.case_feedback_lambda.add_to_role_policy(bedrock_policy)
        self.case_feedback_v2_lambda.add_to_role_policy(bedrock_policy)
        self.case_enrichment_challenge_worker_lambda.add_to_role_policy(bedrock_policy)
        
        self.path_generate_lambda.add_to_role_policy(ssm_policy)
        self.path_generate_worker_lambda.add_to_role_policy(ssm_policy)
        self.path_evaluate_lambda.add_to_role_policy(ssm_policy)
        self.path_evaluate_v2_lambda.add_to_role_policy(ssm_policy)
        self.path_feedback_lambda.add_to_role_policy(ssm_policy)
        self.path_feedback_v2_lambda.add_to_role_policy(ssm_policy)
        self.path_regenerate_challenge_lambda.add_to_role_policy(ssm_policy)
        self.path_regenerate_challenge_v2_lambda.add_to_role_policy(ssm_policy)
        self.case_generate_lambda.add_to_role_policy(ssm_policy)
        self.case_generate_worker_lambda.add_to_role_policy(ssm_policy)
        self.case_generate_path_lambda.add_to_role_policy(ssm_policy)
        self.case_generate_path_worker_lambda.add_to_role_policy(ssm_policy)
        self.case_evaluate_lambda.add_to_role_policy(ssm_policy)
        self.case_evaluate_v2_lambda.add_to_role_policy(ssm_policy)
        self.case_feedback_lambda.add_to_role_policy(ssm_policy)
        self.case_feedback_v2_lambda.add_to_role_policy(ssm_policy)
        self.case_enrichment_challenge_worker_lambda.add_to_role_policy(ssm_policy)

        self.path_generate_lambda.add_to_role_policy(secrets_policy)
        self.path_generate_worker_lambda.add_to_role_policy(secrets_policy)
        
    def create_api_gateway(self):
        """
        Method to create the REST-API Gateway for exposing the main functionalities
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
        root_api = self.api_ruta_estandar.root.add_resource("api")
        root_v1 = root_api.add_resource("v1")
        root_v2 = root_api.add_resource("v2")

        # --- /v1/path endpoints ---
        root_path_v1 = root_v1.add_resource("path")
        path_generate_v1 = root_path_v1.add_resource("generate")
        path_regenerate_challenge_v1 = root_path_v1.add_resource("regenerate_challenge")
        path_evaluate_v1 = root_path_v1.add_resource("evaluate")
        path_feedback_v1 = root_path_v1.add_resource("feedback")

        path_generate_v1.add_method("POST", apigw.LambdaIntegration(self.path_generate_lambda))
        path_regenerate_challenge_v1.add_method("POST", apigw.LambdaIntegration(self.path_regenerate_challenge_lambda))
        path_evaluate_v1.add_method("POST", apigw.LambdaIntegration(self.path_evaluate_lambda))
        path_feedback_v1.add_method("POST", apigw.LambdaIntegration(self.path_feedback_lambda))

        # --- /v1/case endpoints ---
        root_case_v1 = root_v1.add_resource("case")
        case_generate_v1 = root_case_v1.add_resource("generate")
        case_generate_path_v1 = root_case_v1.add_resource("generate_path")
        case_evaluate_v1 = root_case_v1.add_resource("evaluate")
        case_feedback_v1 = root_case_v1.add_resource("feedback")

        case_generate_v1.add_method("POST", apigw.LambdaIntegration(self.case_generate_lambda))
        case_generate_path_v1.add_method("POST", apigw.LambdaIntegration(self.case_generate_path_lambda))
        case_evaluate_v1.add_method("POST", apigw.LambdaIntegration(self.case_evaluate_lambda))
        case_feedback_v1.add_method("POST", apigw.LambdaIntegration(self.case_feedback_lambda))

        # --- /v2/path endpoints ---
        root_path_v2 = root_v2.add_resource("path")
        path_generate_start = root_path_v2.add_resource("generate_start")
        path_regenerate_challenge_v2 = root_path_v2.add_resource("regenerate_challenge")
        path_evaluate_v2 = root_path_v2.add_resource("evaluate")
        path_feedback_v2 = root_path_v2.add_resource("feedback")

        path_generate_start.add_method("POST", apigw.LambdaIntegration(self.path_generate_start_lambda))
        path_regenerate_challenge_v2.add_method("POST", apigw.LambdaIntegration(self.path_regenerate_challenge_v2_lambda))
        path_evaluate_v2.add_method("POST", apigw.LambdaIntegration(self.path_evaluate_v2_lambda))
        path_feedback_v2.add_method("POST", apigw.LambdaIntegration(self.path_feedback_v2_lambda))

        # --- /v2/case endpoints ---
        root_case_v2 = root_v2.add_resource("case")
        case_generate_start = root_case_v2.add_resource("generate_start")
        case_generate_path_start = root_case_v2.add_resource("generate_path_start")
        case_evaluate_v2 = root_case_v2.add_resource("evaluate")
        case_feedback_v2 = root_case_v2.add_resource("feedback")
        case_enrichment_challenge_start = root_case_v2.add_resource("enrichment_challenge_start")

        case_generate_start.add_method("POST", apigw.LambdaIntegration(self.case_generate_start_lambda))
        case_generate_path_start.add_method("POST", apigw.LambdaIntegration(self.case_generate_path_start_lambda))
        case_evaluate_v2.add_method("POST", apigw.LambdaIntegration(self.case_evaluate_v2_lambda))
        case_feedback_v2.add_method("POST", apigw.LambdaIntegration(self.case_feedback_v2_lambda))
        case_enrichment_challenge_start.add_method("POST", apigw.LambdaIntegration(self.case_enrichment_challenge_start_lambda))

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
         