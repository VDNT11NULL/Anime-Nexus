pipeline{
    agent any

    stages{
        stage('Cloning from Github....'){
            steps{
                script{
                    echo "Cloning the repository from GitHub"
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/VDNT11NULL/Anime-Nexus.git']])
                }
            }
        }
    }
}