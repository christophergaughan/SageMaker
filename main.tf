provider "aws" {
  region = "us-east-1"
}

resource "aws_iam_role" "sagemaker_execution_role" {
  name = "sagemaker_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "sagemaker.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_sagemaker_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_sagemaker_notebook_instance" "sagemaker_notebook" {
  name             = "nlp-model-notebook"
  instance_type    = "ml.t2.medium"
  role_arn         = aws_iam_role.sagemaker_execution_role.arn
  lifecycle_config_name = null

  tags = {
    Project = "SageMakerNLP"
    Owner   = "Chris G"
    Env     = "Dev"
  }
}
