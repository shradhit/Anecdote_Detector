from django.conf.urls import url

from myapp.views import init

from . import views

urlpatterns = [
    url(r'^login$', views.loginPage, name='loginPage'),
    url(r'^logout$', views.logout, name='logout'),
    url(r'^register$', views.registerPage, name='registerPage'),
    url(r'^home$', views.home, name='home'),
    url(r'^process$', views.process, name='process'),
    url(r'^download$', views.download, name='download'),
    url(r'^$', views.home, name='home'),
]
init()
