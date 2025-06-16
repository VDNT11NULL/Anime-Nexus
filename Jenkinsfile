pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = 'anime-rec-sys'
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
        KUBECTL_AUTH_PLUGIN = "/usr/lib/google-cloud-sdk/bin"

    }

    stages{
        stage('Cloning from Github....'){
            steps{
                script{
                    echo "Cloning the repository from GitHub"
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/VDNT11NULL/Anime-Nexus.git']])
                }
            }
        }
        
        stage("Making a virtual environment...."){
            steps{
                script{
                    echo 'Making a virtual environment...'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    pip install  dvc
                    '''
                }
            }
        }


        stage('DVC Pull'){
            steps{
                withCredentials([file(credentialsId:'gcp-key' , variable: 'GOOGLE_APPLICATION_CREDENTIALS' )]){
                    script{
                        echo 'DVC Pul....'
                        sh '''
                        . ${VENV_DIR}/bin/activate
                        dvc pull
                        '''
                    }
                }
            }
        }

        stage('Build and Push Image to GCR'){
            steps{
                withCredentials([file(credentialsId:'gcp-key' , variable: 'GOOGLE_APPLICATION_CREDENTIALS' )]){
                    script{
                        echo 'Build and Push Image to GCR'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker --quiet
                        docker build --no-cache -t gcr.io/${GCP_PROJECT}/anime-nexus:latest .
                        docker push gcr.io/${GCP_PROJECT}/anime-nexus:latest
                        '''
                    }
                }
            }
        }

        stage('Deploying to Kubernetes'){
            steps{
                withCredentials([file(credentialsId:'gcp-key' , variable: 'GOOGLE_APPLICATION_CREDENTIALS' )]){
                    script{
                        echo 'Deploying to Kubernetes'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}:${KUBECTL_AUTH_PLUGIN}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud container clusters get-credentials rec-sys-cluster --region us-central1 
                        kubectl apply -f deployment.yaml
                        '''
                    }
                }
            }
        }

    }
}